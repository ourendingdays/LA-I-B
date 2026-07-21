# Notebooks

Every notebook has been build and tested with `CPX22 — 2 vCPUs, 4 GB RAM`. The notebooks are made to be lighweight, fast and eccessible, which means :

- anyone can run the code, even on the not the strongest hardware
- no model is beeing downloaded, as is often the case with hugging face's `transformer`
- when there is a need of a model from HF, the hosted inference API is used (`InferenceClient`) - must have <b>HuggingFace token</b>

Usually people do the following: they download the entire models (weights, config, tokenizer), load it into RAM, and run every calculation on the CPU/GPU, so that their machine does all the work... those rich developers!!!! And I do not want that .. i am a simple gut who just wants to see resutls and not pay for them, so I am using Hugging face's Inference API  heavilly.
What the Inference API does: send your text over the internet to HF's servers (which have beefy GPUs), they run the model, and send back the result. Your machine just sends and receives text.



## Essentials

1. install requirements.txt
``` bash
cd ra-i-g/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Jupyter Notebook Kernel

3. Create HuggingFace Token at `huggingface.co/settings/tokens`
