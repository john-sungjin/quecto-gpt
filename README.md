An atrocious implementation of GPT-2. Goals:
- Refresh my Rust skills (it's been over a year since writing any Rust, and it really shows)
- Feel confidence in my ability to understand transformers

Heavily referenced from https://github.com/jaymody/picoGPT, inspired by https://github.com/newhouseb/potatogpt (though I don't have the wonderful type system).

To get the weights: download `model.safetensors` and `vocab.json` from the [HuggingFace Repo](https://huggingface.co/gpt2/tree/main) and put them in the `weights` directory.