"use client"
import FooterLink from "@/components/forms/FooterLink"
import InputField from "@/components/forms/InputField"
import { Button } from "@/components/ui/button"
import { useForm } from "react-hook-form"

const SignUp = () => {
    const {
        register,
        handleSubmit,
        formState: { errors, isSubmitting },
    } = useForm<SignInFormData>({
        defaultValues: {
            email: "",
            password: "",
        },
        mode: "onBlur"
    })

    const onSubmit = async (data: SignInFormData) => {
        try {
            console.log(data)
        } catch (error) {
            console.error(error)
        }
    }

    return (
        <>
            <h1 className='form-title'>Sign In</h1>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
                <InputField 
                    name="email"
                    label="Email"
                    placeholder="johndoe@gmail.com"
                    register={register}
                    error={errors.email}
                    validation={{required: "Email is required", pattern: /^\w+@\w+\.\w+$/, message: "Invalid email format"}}
                />
                <InputField 
                    name="password"
                    label="Password"
                    placeholder="Enter a strong password"
                    type="password"
                    register={register}
                    error={errors.password}
                    validation={{required: "Password is required", minLength: 8}}
                />
                <Button type="submit" disabled={isSubmitting} className="yellow-btn w-full mt-5">
                    {isSubmitting ? "Logging In..." : "Log In"}
                </Button>
                <FooterLink text="Don't have an account?" linkText="Sign Up" href="/sign-up" />
            </form>
        </>
    )
}

export default SignUp