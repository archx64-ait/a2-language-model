from django import forms

style = forms.Select(attrs={"class": "form-control"})


class NLPForm(forms.Form):
    temperature = forms.ChoiceField(
        choices=[
            (0.5, "0.5"),
            # (0.55, "0.55"),
            # (0.6, "0.6"),
            # (0.65, "0.65"),
            (0.7, "0.7"),
            (0.75, "0.75"),
            (0.8, "0.8"),
            # (0.85, "0.85"),
            # (0.9, "0.9"),
            # (0.95, "0.95"),
            (1, "1"),
        ],
        widget=style,
    )

    prompt = forms.CharField(required=True)
