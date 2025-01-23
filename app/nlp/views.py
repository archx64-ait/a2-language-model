from django.views.generic import TemplateView
from django.views.generic.edit import FormView
from nlp.forms import NLPForm
from typing import Any
from nlp.utils_scratch import *
from django.urls import reverse_lazy
from django.shortcuts import redirect
import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "webapp", "best-val-lstm_lm.pt")


class IndexView(TemplateView):
    template_name = "index.html"


class SuccessView(TemplateView):
    template_name = "success.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        result = self.request.GET.get("result")

        try:
            # Add the result to the context
            context["result"] = result

        except ValueError:
            context["result"] = [""]

        return context


class NLPFormView(FormView):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    form_class = NLPForm
    template_name = "nlp.html"
    model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
    model.load_state_dict(torch.load(MODEL_PATH,  map_location=device))


    def generate(
        self,
        prompt,
        max_seq_len,
        temperature,
        model,
        tokenizer,
        vocab,
        device,
        seed=None,
    ):
        try:
            temperature = float(temperature)
        except ValueError:
            temperature = 0.5

        if seed is not None:
            torch.manual_seed(seed)
            
        model.eval()
        tokens = tokenizer(prompt)
        indices = [vocab[t] for t in tokens]
        batch_size = 1
        hidden = model.init_hidden(batch_size, device)
        with torch.no_grad():
            for _ in range(max_seq_len):
                src = torch.LongTensor([indices]).to(device)
                prediction, hidden = model(src, hidden)

                # Get probabilities for the last word
                probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
                prediction = torch.multinomial(probs, num_samples=1).item()

                # Skip <unk> tokens
                while prediction == vocab["<unk>"]:
                    prediction = torch.multinomial(probs, num_samples=1).item()

                # Stop at <eos> tokens
                if prediction == vocab["<eos>"]:
                    break

                indices.append(prediction)

        itos = vocab.get_itos()
        tokens = [itos[i] for i in indices]
        return tokens

    def generate_sentence(self, model, prompt, temperature):
        # Ensure vocab and tokenizer are properly initialized in your utils
        generation = self.generate(
            prompt=prompt,
            max_seq_len=50,  # Adjust as needed
            temperature=temperature,
            model=model,
            tokenizer=tokenizer,
            vocab=vocab,
            device=self.device,
            seed=seed,
        )
        return " ".join(generation)
            

    def form_valid(self, form):
        # temperature = form.cleaned_data["temperature"]
        # prompt = form.cleaned_data["prompt"]
        # result = self.generate_sentence(model=self.model, prompt=prompt, temperature=temperature)
        # context = self.get_context_data(result=result)
        # # return redirect(f"{reverse_lazy('nlp:success')}?result={result}")
        # return self.render_to_response(context)

        temperature = form.cleaned_data["temperature"]
        prompt = form.cleaned_data["prompt"]
        result = self.generate_sentence(model=self.model, prompt=prompt, temperature=temperature)
        context = self.get_context_data(result=result)
        print(context)
        return self.render_to_response(context)

    def form_invalid(self, form):
        return super().form_invalid(form)

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        # context["results"] = getattr(self, "result", None)
        context["result"] = kwargs.get("result", None)
        return context
