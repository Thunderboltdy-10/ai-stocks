"use client"
import FooterLink from "@/components/forms/FooterLink"
import InputField from "@/components/forms/InputField"
import { Button } from "@/components/ui/button"
import { signInWithEmail } from "@/lib/actions/auth.actions"
import { useRouter } from "next/navigation"
import { useForm } from "react-hook-form"
import { toast } from "sonner"

const SignUp = () => {
    const router = useRouter()
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
            const result = await signInWithEmail(data)
            if (result.success) router.push("/")
        } catch (error) {
            console.error(error)
            toast.error("Sign in failed", {
                description: error instanceof Error ? error.message : "Failed to sign in"
            })
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
                    validation={{required: "Email is required", pattern: /^\S+@\S+\.\S+$/, message: "Invalid email format"}}
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