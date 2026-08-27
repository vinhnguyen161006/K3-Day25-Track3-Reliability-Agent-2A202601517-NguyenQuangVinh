from __future__ import annotations

from dataclasses import dataclass

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker, CircuitOpenError
from reliability_lab.providers import FakeLLMProvider, ProviderError


@dataclass(slots=True)
class GatewayResponse:
    text: str
    route: str
    provider: str | None
    cache_hit: bool
    latency_ms: float
    estimated_cost: float
    error: str | None = None


class ReliabilityGateway:
    """Routes requests through cache, circuit breakers, and fallback providers."""

    def __init__(
        self,
        providers: list[FakeLLMProvider],
        breakers: dict[str, CircuitBreaker],
        cache: ResponseCache | SharedRedisCache | None = None,
    ):
        self.providers = providers
        self.breakers = breakers
        self.cache = cache

    def complete(self, prompt: str) -> GatewayResponse:
        """Route a prompt through cache -> provider chain -> static fallback.

        Cache is checked first; a hit above the similarity threshold short
        circuits the whole pipeline. Otherwise providers are tried in the
        configured order, each guarded by its own circuit breaker, so a
        provider that is open or erroring is skipped rather than retried.
        If every provider fails, a degraded static response is returned
        instead of propagating the error to the caller.
        """
        if self.cache is not None:
            cached, score = self.cache.get(prompt)
            if cached is not None:
                return GatewayResponse(
                    text=cached,
                    route=f"cache_hit:{score:.2f}",
                    provider=None,
                    cache_hit=True,
                    latency_ms=0.0,
                    estimated_cost=0.0,
                )

        last_error: str | None = None
        for index, provider in enumerate(self.providers):
            breaker = self.breakers[provider.name]
            try:
                result = breaker.call(provider.complete, prompt)
            except (ProviderError, CircuitOpenError) as exc:
                last_error = str(exc)
                continue

            if self.cache is not None:
                self.cache.set(prompt, result.text, {"provider": provider.name})
            return GatewayResponse(
                text=result.text,
                route="primary" if index == 0 else "fallback",
                provider=provider.name,
                cache_hit=False,
                latency_ms=result.latency_ms,
                estimated_cost=result.estimated_cost,
            )

        return GatewayResponse(
            text="The service is temporarily degraded. Please try again soon.",
            route="static_fallback",
            provider=None,
            cache_hit=False,
            latency_ms=0.0,
            estimated_cost=0.0,
            error=last_error,
        )
