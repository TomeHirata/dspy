import logging
import random

from pydantic import BaseModel

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.callback import with_callbacks

logger = logging.getLogger(__name__)


class Predict(Module, Parameter):
    def __init__(self, signature, callbacks=None, **config):
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self.callbacks = callbacks or []
        self.reset()

    def reset(self):
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self):
        state_keys = ["traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}

        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()

            for field in demo:
                # FIXME: Saving BaseModels as strings in examples doesn't matter because you never re-access as an object
                demo[field] = serialize_object(demo[field])

            state["demos"].append(demo)

        state["signature"] = self.signature.dump_state()
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state

    def load_state(self, state):
        """Load the saved state of a `Predict` object.

        Args:
            state (dict): The saved state of a `Predict` object.

        Returns:
            self: Returns self to allow method chaining
        """
        excluded_keys = ["signature", "extended_signature", "lm"]
        for name, value in state.items():
            # `excluded_keys` are fields that go through special handling.
            if name not in excluded_keys:
                setattr(self, name, value)

        self.signature = self.signature.load_state(state["signature"])
        self.lm = LM(**state["lm"]) if state["lm"] else None

        if "extended_signature" in state:  # legacy, up to and including 2.5, for CoT.
            raise NotImplementedError("Loading extended_signature is no longer supported in DSPy 2.6+")

        return self

    @with_callbacks
    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    async def acall(self, **kwargs):
        return await self.aforward(**kwargs)

    def _forward_preprocess(self, **kwargs):
        # Extract the three privileged keyword arguments.
        assert "new_signature" not in kwargs, "new_signature is no longer a valid keyword argument."
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or settings.lm
        assert isinstance(lm, BaseLM), "No LM is loaded."

        # If temperature is unset or <=0.15, and n > 1, set temperature to 0.7 to keep randomness.
        temperature = config.get("temperature") or lm.kwargs.get("temperature")
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            logger.warning(
                "Not all input fields were provided to module. Present: %s. Missing: %s.",
                present,
                missing,
            )
        return lm, config, signature, demos, kwargs

    def _forward_postprocess(self, completions, signature, **kwargs):
        pred = Prediction.from_completions(completions, signature=signature)
        if kwargs.pop("_trace", True) and settings.trace is not None:
            trace = settings.trace
            trace.append((self, {**kwargs}, pred))
        return pred

    def forward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        adapter = settings.adapter or ChatAdapter()

        stream_listeners = settings.stream_listeners or []
        stream = settings.send_stream is not None
        if stream and len(stream_listeners) > 0:
            stream = any(stream_listener.predict == self for stream_listener in stream_listeners)

        if stream:
            with settings.context(caller_predict=self):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)
        else:
            with settings.context(send_stream=None):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)

        return self._forward_postprocess(completions, signature, **kwargs)

    async def aforward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        adapter = settings.adapter or ChatAdapter()
        completions = await adapter.acall(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)
        return self._forward_postprocess(completions, signature, **kwargs)

    def update_config(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"


def serialize_object(obj):
    """
    Recursively serialize a given object into a JSON-compatible format.
    Supports Pydantic models, lists, dicts, and primitive types.
    """
    if isinstance(obj, BaseModel):
        # Use model_dump to convert the model into a JSON-serializable dict
        return obj.model_dump()
    elif isinstance(obj, list):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_object(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    else:
        return obj


# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect execution. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.
