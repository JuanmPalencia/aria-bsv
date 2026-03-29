"""aria.integrations — framework-specific helpers for ARIA audit integration.

Available integrations
----------------------

LLM clients
~~~~~~~~~~~
- :mod:`aria.integrations.openai`        — ``ARIAOpenAI``, ``ARIAAsyncOpenAI``
- :mod:`aria.integrations.anthropic`     — ``ARIAAnthropic``, ``ARIAAsyncAnthropic``
- :mod:`aria.integrations.google_gemini` — ``ARIAGemini``, ``ARIAAsyncGemini``
- :mod:`aria.integrations.litellm`       — ``ARIALiteLLM``, ``make_litellm_callback``
- :mod:`aria.integrations.cohere`        — ``ARIACohere``
- :mod:`aria.integrations.mistral`       — ``ARIAMistral``
- :mod:`aria.integrations.ollama`        — ``ARIAOllama``
- :mod:`aria.integrations.vllm`          — ``ARIAvLLM``
- :mod:`aria.integrations.azure_openai`  — ``ARIAAzureOpenAI``

Agent / orchestration frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :mod:`aria.integrations.langchain`       — ``ARIACallbackHandler``, ``ARIAAuditedLLM``
- :mod:`aria.integrations.dspy`            — ``ARIADSPyModule``, ``ARIADSPyOptimizer``
- :mod:`aria.integrations.instructor`      — ``ARIAInstructor``, ``aria_patch``
- :mod:`aria.integrations.smolagents`      — ``ARIASmolAgent``, ``ARIAToolWrapper``
- :mod:`aria.integrations.semantic_kernel` — ``ARIASemanticKernel``, ``ARIAKernelMiddleware``
- :mod:`aria.integrations.crewai`          — ``ARIACrewAI``
- :mod:`aria.integrations.autogen`         — ``ARIAAutogen``
- :mod:`aria.integrations.llamaindex`      — ``ARIALlamaIndex``
- :mod:`aria.integrations.langgraph`       — ``ARIALangGraph``

Structured outputs
~~~~~~~~~~~~~~~~~~
- :mod:`aria.integrations.instructor` — ``ARIAInstructor``, ``aria_patch``

ML platform / experiment tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :mod:`aria.integrations.huggingface` — ``ARIAHuggingFace``
- :mod:`aria.integrations.mlflow`      — ``ARIAMLflow``
- :mod:`aria.integrations.wandb`       — ``ARIAWandB``
- :mod:`aria.integrations.sagemaker`   — ``ARIASageMaker``
- :mod:`aria.integrations.vertexai`    — ``ARIAVertexAI``

Web frameworks
~~~~~~~~~~~~~~
- :mod:`aria.integrations.fastapi` — ``ARIAFastAPI``
- :mod:`aria.integrations.django`  — ``ARIADjango``
- :mod:`aria.integrations.flask`   — ``ARIAFlask``

All integrations handle ``ImportError`` gracefully: if the optional dependency
is not installed, a clear ``ImportError`` with ``pip install aria-bsv[<extra>]``
instructions is raised on instantiation, not at import time.
"""
