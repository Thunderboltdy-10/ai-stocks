import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const { symbol, riskProfile = 'conservative' } = await request.json();

  try {
    const response = await fetch('http://localhost:8000/api/predict-ensemble', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ symbol, risk_profile: riskProfile }),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to get ensemble prediction' },
      { status: 500 }
    );
  }
}
