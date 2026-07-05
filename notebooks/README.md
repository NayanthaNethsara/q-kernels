# LLM Systems Labs — hands-on companion to the research proposal

Eight self-contained Colab notebooks. Each teaches one concept from the proposal
(direction-asymmetric precision + asynchronous pipeline parallelism on consumer GPUs)
and produces at least one figure/number you can show your supervisor.

## Run order and what each proves

| # | Notebook | Concept | Proposal link |
|---|----------|---------|---------------|
| 01 | pytorch_fundamentals | autograd; verify dL/dW = X^T d and dL/dX = d W^T; 16-bytes/param rule | why activations are stored; what crosses the wire; LoRA scoping |
| 02 | run_llm_inference | tokens -> logits -> next token; 12H^2 parameter accounting; model anatomy | background fluency |
| 03 | activation_capture | forward hooks; B x S x H boundary tensor; wire-cost math; outlier problem | the Communication Wall, measured; motivates per-block scales |
| 04 | quantization_lab | build the quantizer; outliers vs block scales; RTN bias vs stochastic rounding; forward cliff | epsilon_fwd measured; core algorithm written |
| 05 | pipeline_split_and_asymmetry | the detach trick = PP on one GPU; gradient-exactness check; fwd-vs-bwd fragility curves | THE asymmetry figure — the thesis's empirical heart |
| 06 | distributed_two_process | torch.distributed send/recv; GPipe micro-batching; tc netem network emulation | the real data path; your programmable 'cluster' |
| 07 | finetune_full_vs_lora | training loop; loss curves; peak-VRAM measurement; LoRA | the actual workload; 'at loss parity' methodology |
| 08 | profiler_and_cost_model | measure T, L_pack, B, alpha; predict vs measure; argmin under constraints; heterogeneity sweep | Objective O1 in miniature; the headline table |

## Environment
- Google Colab, free T4 GPU (`Runtime -> Change runtime type -> T4`). Labs 01/06 also run CPU-only.
- Everything installs in-notebook (`transformers`, `peft`, `datasets`, `matplotlib`).
- Lab 06's netem cells need Linux + sudo (Colab has both). Always run the `tc qdisc del` cleanup cell.

## Working discipline (from the project plan)
- git this folder from day one; commit after every lab.
- Save every figure into `results/` with the exact code that produced it.
- Do not advance a lab while the previous lab's final figure is missing — each phase's bugs hide inside the next phase's complexity.

## Mapping to project phases
Phase 0 = Labs 01-04. Phase 1 = Labs 05-06. Phase 2 = Lab 08 scaled up (real links, real solver).
Phase 3 = rewrite Lab 04's quantizer as a fused Triton kernel. Phase 4 = overlap Lab 06's sends on a
side CUDA stream. Phase 5 = Lab 08's sweep, on the real system, with TAH-Quant as a baseline.
