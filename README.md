# superAGI
# Coding Assignment: Implementation and Optimization of GPT-2 Model (100 points)

---

**Objective:**

This assignment aims to test your understand  ing of the Transformer architecture, and your ability to modify its structures for improved performance. Further, it requires your skills to develop an efficient training loop and implementation of distributed training applicable across multiple GPUs.

**Points:**

Total points for the assignment are **100** and are distributed as follows:

- Task 1: Model Implementation and Checkpoints (20 points)
- Task 2: Architectural changes (40 points)
- Task 3: Distributed Training (40 points)

Please, note that partial points will be awarded on all parts of the assignment, so be sure to clearly communicate your methodologies, insights, and results.

### Task 1 | GPT-2 Model & Checkpoints (20 Points)

---

Start by implementing the `GPT2-small` model (with 125 million parameters) using Python and PyTorch. Make sure you touch upon the key aspects of the model like multi-head self-attention mechanism, feed-forward networks and positional encoding.

Key points:

- Follow the original GPT-2 design of using both token and positional embeddings.
- Implement the transformer layers with multi-head self-attention and point-wise feed-forward network.
- You're required to abstain from using pre-built transformer libraries.

Refer to the GPT-2 paper's architecture descriptions in Sections 1 and 2 for further help. ([GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)). Additionally, a great resource could be Andrej Karpathy’s [nanogpt](https://github.com/karpathy/nanoGPT) repository and the [makemore](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&feature=shared) series.

To validate your implementation, load the original GPT-2 125M model checkpoints and run a sample prediction.

**Deliverable:** Complete Python code featuring the GPT-2 model along with demonstration of appropriate testing to verify its functioning.

### Task 2 | Transformer Architectural Changes (40 Points)

---

In the second task, you are required to add alterations to your original GPT-2 model architecture to experiment and assess the potential of improvements. Here's what you need to do:

- **Rotary Positional Embedding:** Replace the original positional embeddings in the GPT-2 model with Rotary embeddings. You may refer to [Su et. al. RoFormer](https://arxiv.org/pdf/2104.09864.pdf).
- **Group Query Attention:** Equip your model with the Group Query Attention mechanism following the insights from the [Ainslie et. al. GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/pdf/2305.13245v2.pdf). Analyze how this mechanism can modify the model's operation compared to the standard attention mechanism.
- **Sliding Window Attention:** Imbibe the Sliding Window Attention mechanism in your model and observe its effects on model performance. Refer to the work by [Beltagy et. al. Longformer](https://arxiv.org/pdf/2004.05150v2.pdf) for better comprehension of its implementation and advantages.

**Deliverable:** Python code with any one, two or all three changes. Comment on the model size and capabilities, potential pitfalls and/or any improvement after each change. Points will be awarded for any combination of successful implementations.

**Evaluation Scheme:** Each feature implementation will account for:

- Rotary Positional Embedding: 15 points
- Group Query Attention: 10 points
- Sliding Window Attention: 15 points

## Task 3: Training Loop Implementation (40 Points)

---

Finally, create a training loop considering these following requirements:

1. **Single GPU Training Loop:** Your base implementation should be equipped to train your model on a single GPU setup.
2. **Distributed Data Parallel (DDP):** Extend your single GPU training loop to support training across multiple GPUs using DDP. Revisit the [PyTorch's DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for guidance.
3. **Fully Sharded Data Parallel (FSDP):** Implement FSDP as a part of your training loop to shard the model parameters, gradients, and optimizer state. You can follow [Gupta et al., 2020, Training GPT-3 Like Models on a Single Machine](https://arxiv.org/pdf/2101.06840.pdf) for a comprehensive understanding of it.

**Deliverable:** A Python script containing a functional training loop that is compatible with single GPU, DDP, and FSDP options along with a documentation illustrating how the code adapts to each setting.

**Evaluation Scheme:** Each feature implementation will account for:

- Single GPU: 10 points
- DDP: 10 points
- FSDP: 20 points

**Note:** Document your code, approaches, difficulties encountered, and your solutions 
thoroughly. Include any reference materials you used in your report. Focus on clear communication of your methodologies and results.

**Submission:**

For each subtask, submit your source code and a brief description of your implementations. If relevant, please support your findings with visualizations of the alterations and their impacts.

Please remember, partial points will be awarded for each part, so it's better to submit an incomplete assignment than no assignment at all.
