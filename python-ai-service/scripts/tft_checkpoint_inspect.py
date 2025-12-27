"""
Diagnostic script: inspect TFT checkpoint + config, prepare cached TFT data,
attempt to load model and run a small predict. Prints diagnostic summaries.
Run from repository root with the project's venv Python.
"""
import sys
import os
import json
import traceback
from pathlib import Path
import pprint
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pp = pprint.PrettyPrinter(indent=2)

# optional imports checked at runtime
try:
    import torch
    import pytorch_forecasting as pf
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    import torch.utils.data as tud
    PF_AVAILABLE = True
except Exception:
    PF_AVAILABLE = False

def load_checkpoint(path):
    import torch
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(str(path), map_location='cpu')
    print("Top-level type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
    return ckpt


def extract_state_dict(ckpt):
    # Multiple checkpoint layouts exist (Lightning, plain state_dict, nested)
    candidates = []
    if isinstance(ckpt, dict):
        # common lightning names
        for name in ['state_dict', 'model_state_dict', 'model']:
            if name in ckpt and isinstance(ckpt[name], dict):
                candidates.append((name, ckpt[name]))
        # some checkpoints wrap model under 'checkpoint' or 'pl_state_dict'
        for name in ['checkpoint', 'pl_state_dict', 'net']:
            if name in ckpt and isinstance(ckpt[name], dict):
                # try deeper
                v = ckpt[name]
                for k in ['state_dict', 'model_state_dict', 'model']:
                    if k in v and isinstance(v[k], dict):
                        candidates.append((f"{name}.{k}", v[k]))
        # fallback: any dict-like value with many tensor-like values
        if not candidates:
            # heuristics: pick largest dict-valued entry
            best = None
            best_len = 0
            for k,v in ckpt.items():
                if isinstance(v, dict) and len(v) > best_len:
                    best = (k,v)
                    best_len = len(v)
            if best is not None:
                candidates.append(best)
    else:
        return None, None
    # return the first candidate
    if candidates:
        name, sd = candidates[0]
        print(f"Using state_dict candidate: {name} (len={len(sd)})")
        return name, sd
    return None, None


def print_state_keys(sd, max_print=80):
    import torch
    keys = list(sd.keys())
    print(f"State-dict keys count: {len(keys)}")
    sample = keys[:max_print]
    for k in sample:
        v = sd[k]
        typ = type(v)
        shape = None
        try:
            if hasattr(v, 'shape'):
                shape = tuple(v.shape)
        except Exception:
            shape = None
        print(f"  {k} -> {typ.__name__}, shape={shape}")


def read_config_candidates():
    # look for helpful JSON config files near saved_models
    cand_paths = []
    sm = ROOT / 'saved_models'
    if sm.exists():
        # saved_models/tft/AAPL/*.json
        p1 = sm / 'tft' / 'AAPL'
        if p1.exists():
            for j in p1.glob('*.json'):
                cand_paths.append(j)
        # saved_models/AAPL_tft_config.json or saved_models/AAPL_tft_config*.json
        for j in sm.glob('AAPL*config*.json'):
            cand_paths.append(j)
        # any JSON in saved_models root
        for j in sm.glob('*.json'):
            cand_paths.append(j)
    return cand_paths


def try_prepare_cached_df():
    try:
        import inference.predict_tft as pmod
        print("Calling prepare_inference_data('AAPL', use_cache_only=True, include_sentiment=True) ...")
        df = pmod.prepare_inference_data('AAPL', use_cache_only=True, include_sentiment=True)
        print("Prepared DF shape:", getattr(df, 'shape', None))
        print("Columns sample (first 40):", list(df.columns)[:40])
        print("DF head:")
        print(df.head(3).to_string())
        return df
    except Exception as e:
        print("prepare_inference_data failed:")
        traceback.print_exc()
        return None


def try_load_model_and_predict(ckpt_path, df):
    try:
        import inference.predict_tft as pmod
        print("Calling load_tft_model(...)")
        model_info = pmod.load_tft_model(str(ckpt_path), symbol='AAPL', map_location='cpu')
        print("load_tft_model returned:", type(model_info))
        # model_info may be (model, extras) or model
        model = None
        extras = None
        if isinstance(model_info, tuple) and len(model_info) >= 1:
            model = model_info[0]
            extras = model_info[1:] if len(model_info) > 1 else None
        else:
            model = model_info
        print("Model type:", type(model))
        if hasattr(model, 'to'):
            try:
                model.to('cpu')
            except Exception:
                pass
        # attempt forecast via existing helper first
        print("Attempting generate_forecast(model, df, horizon=5)")
        try:
            forecast = pmod.generate_forecast(model, df, horizon=5)
            print("Forecast result type:", type(forecast))
            if isinstance(forecast, dict):
                print("Forecast keys:", list(forecast.keys()))
                if 'predictions' in forecast:
                    preds = forecast['predictions']
                    print("Predictions sample shape/type:", type(preds), getattr(preds, 'shape', None))
            else:
                pp.pprint(forecast)
        except Exception:
            print("generate_forecast raised an exception:")
            traceback.print_exc()

        # If pytorch-forecasting is available, try reconstructing TimeSeriesDataSet
        if not PF_AVAILABLE:
            print('\nPyTorch-Forecasting not available in venv; skipping dataset reconstruction.')
            return

        try:
            # attempt to build dataset from saved config
            cfg_path = ROOT / 'saved_models' / 'AAPL_tft_config.json'
            if not cfg_path.exists():
                print('Config JSON not found at', cfg_path)
                return
            cfg = json.load(open(cfg_path,'r'))
            print('\nReconstructing TimeSeriesDataSet using', cfg_path)
            max_encoder_length = int(cfg.get('max_encoder_length', 60))
            max_prediction_length = int(cfg.get('max_prediction_length', 10))
            group_ids = cfg.get('group_ids', ['symbol'])
            target = cfg.get('target', 'returns')
            static_categoricals = cfg.get('static_categoricals', [])
            static_reals = cfg.get('static_reals', [])
            time_varying_known_categoricals = cfg.get('time_varying_known_categoricals', [])
            time_varying_known_reals = cfg.get('time_varying_known_reals', [])
            time_varying_unknown_reals = cfg.get('time_varying_unknown_reals', [])

            # ensure group id and static fields exist in df
            df2 = df.copy()
            for gid in group_ids:
                if gid not in df2.columns:
                    df2[gid] = 'AAPL'
            for sc in static_categoricals:
                if sc not in df2.columns:
                    df2[sc] = 'N/A'
            for sr in static_reals:
                if sr not in df2.columns:
                    df2[sr] = 0.0

            # derive date-based known categoricals if missing
            if 'Date' in df2.columns:
                try:
                    df2['Date'] = pd.to_datetime(df2['Date'])
                except Exception:
                    pass
            # day_of_week, month, quarter are commonly required
            if 'day_of_week' in time_varying_known_categoricals and 'day_of_week' not in df2.columns:
                if 'Date' in df2.columns:
                    df2['day_of_week'] = df2['Date'].dt.dayofweek
                else:
                    df2['day_of_week'] = 0
            if 'month' in time_varying_known_categoricals and 'month' not in df2.columns:
                if 'Date' in df2.columns:
                    df2['month'] = df2['Date'].dt.month
                else:
                    df2['month'] = 1
            if 'quarter' in time_varying_known_categoricals and 'quarter' not in df2.columns:
                if 'Date' in df2.columns:
                    df2['quarter'] = df2['Date'].dt.quarter
                else:
                    df2['quarter'] = 1

            # ensure categorical columns are of string type for safety
            for cat in (static_categoricals + time_varying_known_categoricals):
                if cat in df2.columns:
                    df2[cat] = df2[cat].astype(str)

            # ensure known/unknown real columns exist (fill with zeros when missing)
            for real_col in list(time_varying_known_reals) + list(time_varying_unknown_reals):
                if real_col not in df2.columns:
                    df2[real_col] = 0.0

            # Enforce exact feature set: add any missing required cols and drop extras
            required_cols = set(
                [
                    'time_idx',
                    'Date',
                    target,
                ]
            )
            required_cols.update(group_ids)
            required_cols.update(static_categoricals)
            required_cols.update(static_reals)
            required_cols.update(time_varying_known_categoricals)
            required_cols.update(time_varying_known_reals)
            required_cols.update(time_varying_unknown_reals)

            # Add any missing required columns with sensible defaults
            missing = sorted(list(required_cols - set(df2.columns)))
            if missing:
                print('Missing required columns (will be added with defaults):', missing)
            for c in missing:
                if c in (static_categoricals + time_varying_known_categoricals):
                    df2[c] = 'N/A'
                else:
                    df2[c] = 0.0

            # Drop any noisy / extra columns that are not part of the required set
            extras = sorted([c for c in df2.columns if c not in required_cols and c != 'Date'])
            if extras:
                print('Dropping extra columns to match checkpoint feature set (count=%d)...' % len(extras))
                # keep a small sample of extras for debugging
                print('Extra columns sample:', extras[:20])
                df2 = df2[[c for c in df2.columns if c in required_cols or c == 'Date']]

            print('After enforcement, DF shape:', df2.shape)

            # build TimeSeriesDataSet
            print('Building TimeSeriesDataSet with encoder/prediction lengths:', max_encoder_length, max_prediction_length)
            dataset = TimeSeriesDataSet(
                df2,
                time_idx='time_idx',
                target=target,
                group_ids=group_ids,
                min_encoder_length=max_encoder_length,
                max_encoder_length=max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_known_categoricals=time_varying_known_categoricals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                target_normalizer=GroupNormalizer(groups=group_ids)
            )

            print('Dataset built. Examples:', len(dataset))
            # create dataloader
            batch_size = int(cfg.get('batch_size', 64))
            dl = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
            print('Dataloader length (batches):', len(dl))

            # Try constructing a matching TFT via from_dataset and loading checkpoint state_dict into it
            try:
                print('\nAttempting to instantiate TFT from dataset and load checkpoint state_dict (non-strict)')
                from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
                import torch

                tft_kwargs = dict(
                    hidden_size=int(cfg.get('hidden_size', 128)),
                    attention_head_size=int(cfg.get('attention_head_size', 4)),
                    dropout=float(cfg.get('dropout', 0.3)),
                    hidden_continuous_size=int(cfg.get('hidden_continuous_size', 64)),
                    output_size=int(cfg.get('output_size', 7)),
                    learning_rate=float(cfg.get('learning_rate', 0.001)),
                )
                print('TFT kwargs:', tft_kwargs)
                model_from_ds = TemporalFusionTransformer.from_dataset(dataset, **tft_kwargs)
                print('Instantiated TFT from dataset; now loading checkpoint state_dict...')
                ck = torch.load(str(ckpt_path), map_location='cpu')
                sd_candidate = None
                if isinstance(ck, dict):
                    for name in ('state_dict','model_state_dict','model'):
                        if name in ck and isinstance(ck[name], dict):
                            sd_candidate = ck[name]
                            break
                    if sd_candidate is None:
                        # fallback to largest dict
                        best = None
                        best_len = 0
                        for k,v in ck.items():
                            if isinstance(v, dict) and len(v) > best_len:
                                best = v
                                best_len = len(v)
                        sd_candidate = best
                if sd_candidate is None:
                    print('No state-dict found in checkpoint for loading into model_from_ds')
                else:
                    # strip 'model.' prefix when present
                    new_state = { (k[6:] if k.startswith('model.') else k): v for k,v in sd_candidate.items() }
                    try:
                        res = model_from_ds.load_state_dict(new_state, strict=False)
                        print('load_state_dict result:', res)
                    except RuntimeError as e:
                        print('load_state_dict raised RuntimeError (likely size mismatches).')
                        print('Exception message:')
                        print(str(e))
                        # also print a short summary of mismatches for easier triage
                        msg = str(e)
                        # print first 1200 chars to avoid overwhelming output
                        print(msg[:1200])
                    # Second strategy: only copy keys whose shapes exactly match the target model
                    try:
                        print('\nAttempting conservative shape-matching parameter copy (only exact-shape keys)')
                        target_state = model_from_ds.state_dict()
                        matched = {}
                        for k, v in sd_candidate.items():
                            k2 = (k[6:] if k.startswith('model.') else k)
                            if k2 in target_state:
                                tv = target_state[k2]
                                try:
                                    if hasattr(v, 'shape') and hasattr(tv, 'shape') and tuple(v.shape) == tuple(tv.shape):
                                        matched[k2] = v
                                except Exception:
                                    pass
                        print('Exact-shape matched keys count:', len(matched))
                        if matched:
                            # load these safer matched params
                            res2 = model_from_ds.load_state_dict(matched, strict=False)
                            print('Conservative load_state_dict result:', res2)
                        else:
                            print('No exact-shape matches found; skipping conservative load.')
                    except Exception:
                        print('Conservative matching load failed:')
                        traceback.print_exc()
                    try:
                        print('Now trying predict with model_from_ds...')
                        preds2 = model_from_ds.predict(dl)
                        print('model_from_ds.predict returned type:', type(preds2))
                        print('Sample shape/info:', getattr(preds2, 'shape', None))
                    except Exception:
                        print('model_from_ds.predict raised:')
                        traceback.print_exc()
            except Exception:
                print('TFT from_dataset + load attempt failed:')
                traceback.print_exc()

            print('Calling model.predict(dataloader) ...')
            # inspect model loss and hyperparameters before predicting
            try:
                print('\nModel hyperparameters (hparams):')
                try:
                    pp.pprint(dict(model.hparams))
                except Exception:
                    try:
                        print(model.hparams)
                    except Exception:
                        pass
                print('\nModel.loss:', getattr(model, 'loss', None))
                print('Type of model.loss:', type(getattr(model, 'loss', None)))
                if getattr(model, 'loss', None) is None:
                    print('\nWarning: model.loss is None — prediction likely to fail due to missing normalizer/encoder')
                else:
                    try:
                        print('model.loss attributes:', [a for a in dir(model.loss) if not a.startswith('_')][:40])
                    except Exception:
                        pass

                preds = model.predict(dl)
                print('model.predict returned type:', type(preds))
                try:
                    print('Sample prediction shape/info:', getattr(preds, 'shape', None))
                except Exception:
                    pass
            except Exception:
                print('model.predict(dataloader) raised:')
                traceback.print_exc()
        except Exception:
            print('Reconstruction/predict attempt failed:')
            traceback.print_exc()
    except Exception:
        print("load_tft_model or predict failed:")
        traceback.print_exc()


if __name__ == '__main__':
    try:
        ckpt_path = ROOT / 'saved_models' / 'tft' / 'AAPL' / 'best_model.ckpt'
        if not ckpt_path.exists():
            alt = ROOT / 'saved_models' / 'AAPL_tft.ckpt'
            if alt.exists():
                ckpt_path = alt
        if not ckpt_path.exists():
            print("Checkpoint not found at expected locations:", ckpt_path)
            sys.exit(2)

        ckpt = load_checkpoint(ckpt_path)
        sd_name, sd = extract_state_dict(ckpt)
        if sd is None:
            print("No state-dict-like object found in checkpoint; printing full keys")
            if isinstance(ckpt, dict):
                for k,v in ckpt.items():
                    print(f" - {k}: type={type(v)}")
        else:
            print_state_keys(sd, max_print=120)

        # config candidates
        configs = read_config_candidates()
        print("Config candidate files found:", configs)
        for c in configs:
            try:
                print('\n---', c)
                j = json.load(open(c,'r'))
                pp.pprint(j)
            except Exception:
                print("Failed reading config file:", c)
                traceback.print_exc()

        # prepare cached df
        df = try_prepare_cached_df()

        # try load model and predict
        try_load_model_and_predict(ckpt_path, df)

        print('\nDONE')
    except Exception:
        traceback.print_exc()
        sys.exit(3)
