# DeepSeek-V4 toy config

Minimal DeepSeek-V4 text/MoE config for patchgen and loader tests. It keeps the
V4-specific pieces active (Hash-MoE bootstrap, learned-routed MoE, CSA/HCA
compressor layers, mHC streams) while shrinking hidden sizes, experts, vocab and
sequence limits enough for CPU/GPU unit tests.
