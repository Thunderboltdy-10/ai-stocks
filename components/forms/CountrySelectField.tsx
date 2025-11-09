import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { Label } from "../ui/label"
import { useMemo, useState } from "react"
import countryList from 'react-select-country-list'
import { Controller } from 'react-hook-form'


const CountrySelectField = ({name, label, placeholder, control, error, required = false} : CountrySelectProps) => {
    const [search, setSearch] = useState("")
    const options = useMemo<{label: string, value: string}[] | undefined>(() => countryList().getData(), [])

    return (
        <div className='space-y-2'>
            <Label htmlFor={name} className='form-label'>{label}</Label>
            <Controller
                name={name}
                control={control}
                rules={{
                    required: required ? `Please select ${label.toLowerCase()}` : false
                }}
                render={({field}) => {
                    const displayedValue =
                        options?.find((option) => option.value === field.value)?.label || search


                    return (
                        <>
                            <Command>
                                <CommandInput 
                                    showIcon={false} 
                                    value={displayedValue}
                                    onValueChange={(value) => {
                                        setSearch(value)
                                        if (field.value && value!== field.value) {
                                            field.onChange(null)
                                        }
                                    }}
                                    placeholder={placeholder}
                                    wrapperClassName="h-12 px-3 border border-gray-600 bg-gray-800 rounded-lg focus-within:border-yellow-500 transition-colors"
                                    className="text-white text-base placeholder:text-gray-500 bg-transparent"
                                />
                                {search.length > 0 && !field.value && <CommandList className="bg-gray-800">
                                    <CommandEmpty className="bg-gray-800 py-2 pl-3 text-sm">No country found</CommandEmpty>
                                    <CommandGroup className="bg-gray-800 max-h-[200px]">
                                        {options && options.filter((option) => option.label.toLowerCase().includes(search.toLowerCase())).map((option) => (
                                            <CommandItem
                                                key={option.value}
                                                value={option.label}
                                                className='hover:bg-gray-600 data-[selected=true]:bg-gray-600'
                                                onSelect={(selectedLabel) => {
                                                    const selectedOption = options?.find(
                                                        (opt) => opt.label === selectedLabel
                                                    )
                                                    if (selectedOption) {
                                                        field.onChange(selectedOption.value)
                                                        setSearch("")
                                                    }
                                                }}

                                            >
                                                {option.label}
                                            </CommandItem>
                                        ))}
                                    </CommandGroup>
                                </CommandList>}
                            </Command>
                            {error && <p className='text-sm text-red-500'>{error.message}</p>}
                        </>
                    )
                }}
            />
        </div>
    )
}

export default CountrySelectField