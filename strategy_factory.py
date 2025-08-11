import logging
from typing import Dict, Any, List, Type

logger = logging.getLogger(__name__)

class StrategyFactory:
    """
    Factory for creating strategy instances.
    """
    _strategies = {}

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type):
        """
        Register a strategy class.
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create_strategy(cls, name: str, params: Dict[str, Any]):
        """
        Create a strategy instance.
        """
        if name not in cls._strategies:
            try:
                from strategies.supertrend import SupertrendStrategy
                cls.register_strategy("supertrend", SupertrendStrategy)
            except ImportError as e:
                logger.error(f"Could not import SupertrendStrategy: {str(e)}")
                raise ValueError(f"Strategy not found: {name}")

        strategy_class = cls._strategies[name]
        return strategy_class(**params)
    
    @classmethod
    def discover_strategies(cls) -> List[str]:
        """
        Return list of available strategies.
        """
        if "supertrend" not in cls._strategies:
            try:
                from strategies.supertrend import SupertrendStrategy
                cls.register_strategy("supertrend", SupertrendStrategy)
            except ImportError:
                pass
        return list(cls._strategies.keys()) or ["supertrend"]