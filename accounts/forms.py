from django import forms

class RegistrationForm(forms.Form):

    role = forms.CharField()

    first_name = forms.CharField(max_length=100)
    last_name = forms.CharField(max_length=100)

    email = forms.EmailField()
    mobile = forms.CharField(max_length=15)

    dob = forms.DateField()

    gender = forms.CharField()

    occupation = forms.CharField(required=False)

    monthly_income = forms.DecimalField(required=False)

    password = forms.CharField(
        widget=forms.PasswordInput
    )

    trusted_contact_name = forms.CharField(
        required=False
    )

    trusted_contact_phone = forms.CharField(
        required=False
    )

    relationship = forms.CharField(
        required=False
    )

    captcha_answer = forms.IntegerField()