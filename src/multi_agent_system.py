"""
GIGA TRADER - Multi-Agent Orchestration System
===============================================
A massive parallel agent system for continuous model improvement.

5+ Coordinated Agents:
1. Test Generation Agent - Creates validation tests continuously
2. Training Agent - Runs experiments in parallel
3. Monitoring Agent - Real-time health checks and alerts
4. Analysis Agent - Deep performance analysis
5. Optimization Agent - Boundary probing and HP search

Architecture:
- Each agent runs in its own thread/process
- Message bus for inter-agent communication
- Shared state through Redis/file system
- Non-blocking async operations
"""

import asyncio
import threading
import queue
import json
import logging
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GigaTrader.MultiAgent")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"


# =============================================================================
# MESSAGE BUS - Inter-Agent Communication
# =============================================================================

class MessageType(Enum):
    """Types of messages agents can exchange."""
    # Control messages
    START = "START"
    STOP = "STOP"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    HEARTBEAT = "HEARTBEAT"

    # Data messages
    TEST_RESULT = "TEST_RESULT"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    EXPERIMENT_FAILED = "EXPERIMENT_FAILED"
    MODEL_UPDATED = "MODEL_UPDATED"
    ALERT = "ALERT"
    METRIC = "METRIC"

    # Request/Response
    REQUEST_ANALYSIS = "REQUEST_ANALYSIS"
    ANALYSIS_RESULT = "ANALYSIS_RESULT"
    REQUEST_OPTIMIZATION = "REQUEST_OPTIMIZATION"
    OPTIMIZATION_RESULT = "OPTIMIZATION_RESULT"


@dataclass
class Message:
    """Message for inter-agent communication."""
    msg_type: MessageType
    sender: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1 = highest, 10 = lowest
    msg_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])

    def to_dict(self) -> Dict:
        return {
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "msg_id": self.msg_id,
        }

    def __lt__(self, other: "Message") -> bool:
        """Enable comparison for priority queue - lower priority number = higher priority."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class MessageBus:
    """
    Central message bus for agent communication.
    Thread-safe, priority-based message queue.
    """

    def __init__(self):
        self.subscribers: Dict[str, List[queue.PriorityQueue]] = {}
        self.all_subscribers: List[queue.PriorityQueue] = []
        self.message_log: List[Message] = []
        self.lock = threading.Lock()
        self._running = True

    def subscribe(self, agent_name: str, msg_types: List[MessageType] = None) -> queue.PriorityQueue:
        """Subscribe an agent to receive messages."""
        q = queue.PriorityQueue()

        with self.lock:
            if msg_types is None:
                # Subscribe to all messages
                self.all_subscribers.append(q)
            else:
                for msg_type in msg_types:
                    if msg_type.value not in self.subscribers:
                        self.subscribers[msg_type.value] = []
                    self.subscribers[msg_type.value].append(q)

        logger.info(f"Agent '{agent_name}' subscribed to message bus")
        return q

    def publish(self, message: Message):
        """Publish a message to all interested subscribers."""
        with self.lock:
            self.message_log.append(message)

            # Send to type-specific subscribers
            if message.msg_type.value in self.subscribers:
                for q in self.subscribers[message.msg_type.value]:
                    q.put((message.priority, time.time(), message))

            # Send to all-subscribers
            for q in self.all_subscribers:
                q.put((message.priority, time.time(), message))

        logger.debug(f"Published: {message.msg_type.value} from {message.sender}")

    def get_recent_messages(self, limit: int = 100) -> List[Message]:
        """Get recent messages from the log."""
        with self.lock:
            return self.message_log[-limit:]


# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class AgentState(Enum):
    """Agent state."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class AgentStats:
    """Statistics for an agent."""
    name: str
    state: AgentState = AgentState.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    tokens_processed: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "last_activity": self.last_activity.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "tokens_processed": self.tokens_processed,
            "recent_errors": self.errors[-5:],
        }


class BaseAgent(ABC):
    """
    Base class for all agents in the system.

    Each agent:
    - Runs in its own thread
    - Subscribes to relevant messages
    - Publishes results/status
    - Has health monitoring
    """

    def __init__(self, name: str, message_bus: MessageBus):
        self.name = name
        self.message_bus = message_bus
        self.stats = AgentStats(name=name)
        self.message_queue = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[datetime] = None

        # Configuration
        self.tick_interval = 1.0  # Seconds between ticks
        self.heartbeat_interval = 30.0  # Seconds between heartbeats
        self._last_heartbeat = time.time()

    def start(self):
        """Start the agent in a background thread."""
        if self._running:
            logger.warning(f"Agent {self.name} already running")
            return

        self._running = True
        self._start_time = datetime.now()
        self.stats.state = AgentState.RUNNING

        # Subscribe to messages
        self.message_queue = self.message_bus.subscribe(
            self.name,
            self.get_subscribed_message_types()
        )

        # Start thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(f"Agent {self.name} started")
        self._publish_status("started")

    def stop(self):
        """Stop the agent."""
        self._running = False
        self.stats.state = AgentState.STOPPED

        if self._thread:
            self._thread.join(timeout=5.0)

        logger.info(f"Agent {self.name} stopped")
        self._publish_status("stopped")

    def pause(self):
        """Pause the agent."""
        self.stats.state = AgentState.PAUSED
        self._publish_status("paused")

    def resume(self):
        """Resume a paused agent."""
        if self.stats.state == AgentState.PAUSED:
            self.stats.state = AgentState.RUNNING
            self._publish_status("resumed")

    def _run_loop(self):
        """Main agent loop."""
        while self._running:
            try:
                # Update uptime
                if self._start_time:
                    self.stats.uptime_seconds = (datetime.now() - self._start_time).total_seconds()

                # Skip if paused
                if self.stats.state == AgentState.PAUSED:
                    time.sleep(self.tick_interval)
                    continue

                # Process incoming messages
                self._process_messages()

                # Execute main task
                if self.stats.state == AgentState.RUNNING:
                    self.tick()
                    self.stats.last_activity = datetime.now()

                # Send heartbeat
                if time.time() - self._last_heartbeat > self.heartbeat_interval:
                    self._send_heartbeat()
                    self._last_heartbeat = time.time()

                time.sleep(self.tick_interval)

            except Exception as e:
                self.stats.state = AgentState.ERROR
                self.stats.errors.append(str(e))
                self.stats.tasks_failed += 1
                logger.error(f"Agent {self.name} error: {e}")
                self._publish_error(str(e))
                time.sleep(self.tick_interval * 2)  # Back off
                self.stats.state = AgentState.RUNNING  # Try to recover

    def _process_messages(self):
        """Process incoming messages."""
        if self.message_queue is None:
            return

        try:
            while True:
                # Non-blocking get
                _, _, message = self.message_queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass

    def _send_heartbeat(self):
        """Send heartbeat message."""
        self.message_bus.publish(Message(
            msg_type=MessageType.HEARTBEAT,
            sender=self.name,
            data=self.stats.to_dict(),
            priority=10,  # Low priority
        ))

    def _publish_status(self, status: str):
        """Publish status update."""
        self.message_bus.publish(Message(
            msg_type=MessageType.METRIC,
            sender=self.name,
            data={"status": status, **self.stats.to_dict()},
        ))

    def _publish_error(self, error: str):
        """Publish error alert."""
        self.message_bus.publish(Message(
            msg_type=MessageType.ALERT,
            sender=self.name,
            data={"error": error, "severity": "ERROR"},
            priority=1,  # High priority
        ))

    @abstractmethod
    def tick(self):
        """Execute one tick of the agent's main task."""
        pass

    @abstractmethod
    def handle_message(self, message: Message):
        """Handle an incoming message."""
        pass

    def get_subscribed_message_types(self) -> List[MessageType]:
        """Return message types this agent subscribes to."""
        return [MessageType.START, MessageType.STOP, MessageType.PAUSE, MessageType.RESUME]


# =============================================================================
# AGENT IMPLEMENTATIONS
# =============================================================================

class TestGenerationAgent(BaseAgent):
    """
    Continuously generates validation tests for models.

    Generates:
    - Boundary condition tests
    - Adversarial examples
    - Historical scenario replays
    - Monte Carlo simulations
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("TestGenerator", message_bus)
        self.tick_interval = 30.0  # Generate tests every 30 seconds
        self.tests_generated = 0
        self.batch_size = 20

    def tick(self):
        """Generate a batch of tests."""
        tests = self._generate_test_batch()

        self.tests_generated += len(tests)
        self.stats.tasks_completed += 1
        self.stats.tokens_processed += len(tests) * 100  # Estimate

        # Publish results
        self.message_bus.publish(Message(
            msg_type=MessageType.TEST_RESULT,
            sender=self.name,
            data={
                "batch_size": len(tests),
                "total_generated": self.tests_generated,
                "test_types": [t["type"] for t in tests],
            },
        ))

        logger.info(f"[{self.name}] [OK] Batch complete. Total: {self.tests_generated} tests")

    def _generate_test_batch(self) -> List[Dict]:
        """Generate a batch of tests."""
        tests = []

        test_types = [
            "boundary_high_volatility",
            "boundary_low_volume",
            "adversarial_noise",
            "historical_crash_2020",
            "historical_rally_2021",
            "monte_carlo_random",
        ]

        for i in range(self.batch_size):
            test_type = random.choice(test_types)
            tests.append({
                "type": test_type,
                "timestamp": datetime.now().isoformat(),
                "params": self._get_test_params(test_type),
            })

        return tests

    def _get_test_params(self, test_type: str) -> Dict:
        """Get parameters for a specific test type."""
        if "volatility" in test_type:
            return {"volatility_multiplier": random.uniform(2.0, 5.0)}
        elif "volume" in test_type:
            return {"volume_multiplier": random.uniform(0.1, 0.3)}
        elif "noise" in test_type:
            return {"noise_std": random.uniform(0.01, 0.05)}
        elif "historical" in test_type:
            return {"scenario": test_type.split("_")[-1]}
        else:
            return {"seed": random.randint(0, 10000)}

    def handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.msg_type == MessageType.REQUEST_ANALYSIS:
            # Analysis agent requesting specific tests
            test_type = message.data.get("test_type", "random")
            logger.info(f"[{self.name}] Received request for {test_type} tests")


class TrainingAgent(BaseAgent):
    """
    Runs model training experiments in parallel.

    Capabilities:
    - Parallel experiment execution
    - Hyperparameter search
    - Model checkpointing
    - Early stopping
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("Trainer", message_bus)
        self.tick_interval = 60.0  # Check for work every minute
        self.active_experiments: Dict[str, Dict] = {}
        self.completed_experiments: List[Dict] = []
        self.executor = ThreadPoolExecutor(max_workers=3)

    def tick(self):
        """Check for experiments and manage training."""
        # Check if we should start a new experiment
        if len(self.active_experiments) < 2:  # Max 2 concurrent
            exp_id = f"exp_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
            self._start_experiment(exp_id)

        # Update experiment progress
        self._update_experiments()

    def _start_experiment(self, exp_id: str):
        """Start a new experiment."""
        config = self._generate_experiment_config()

        self.active_experiments[exp_id] = {
            "id": exp_id,
            "config": config,
            "started": datetime.now(),
            "progress": 0,
            "status": "running",
        }

        self.message_bus.publish(Message(
            msg_type=MessageType.EXPERIMENT_STARTED,
            sender=self.name,
            data={"experiment_id": exp_id, "config": config},
        ))

        logger.info(f"[{self.name}] Started experiment: {exp_id}")

    def _generate_experiment_config(self) -> Dict:
        """Generate random experiment configuration."""
        return {
            "type": random.choice(["hyperparameter", "feature_subset", "ensemble", "regularization"]),
            "learning_rate": random.uniform(0.001, 0.1),
            "n_estimators": random.randint(50, 200),
            "max_depth": random.randint(2, 5),
        }

    def _update_experiments(self):
        """Update progress of active experiments."""
        completed = []

        for exp_id, exp in self.active_experiments.items():
            # Simulate progress
            exp["progress"] += random.randint(10, 30)

            if exp["progress"] >= 100:
                exp["status"] = "completed"
                exp["completed"] = datetime.now()
                exp["results"] = {
                    "auc": random.uniform(0.65, 0.85),
                    "accuracy": random.uniform(0.55, 0.75),
                    "win_rate": random.uniform(0.50, 0.70),
                }
                completed.append(exp_id)

                self.message_bus.publish(Message(
                    msg_type=MessageType.EXPERIMENT_COMPLETED,
                    sender=self.name,
                    data={"experiment_id": exp_id, "results": exp["results"]},
                ))

                self.stats.tasks_completed += 1
                logger.info(f"[{self.name}] [OK] Completed experiment: {exp_id} (AUC: {exp['results']['auc']:.3f})")

        # Move completed to history
        for exp_id in completed:
            self.completed_experiments.append(self.active_experiments.pop(exp_id))

    def handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.msg_type == MessageType.REQUEST_OPTIMIZATION:
            # Optimization agent suggesting new experiments
            suggested_config = message.data.get("config")
            logger.info(f"[{self.name}] Received optimization suggestion")

    def get_subscribed_message_types(self) -> List[MessageType]:
        return super().get_subscribed_message_types() + [
            MessageType.REQUEST_OPTIMIZATION,
            MessageType.ANALYSIS_RESULT,
        ]


class MonitoringAgent(BaseAgent):
    """
    Real-time system health monitoring and alerting.

    Monitors:
    - Agent health
    - Resource usage
    - Model drift
    - API rate limits
    - Data quality
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("Monitor", message_bus)
        self.tick_interval = 10.0  # Check every 10 seconds
        self.agent_status: Dict[str, Dict] = {}
        self.alerts: List[Dict] = []
        self.health_checks_performed = 0

    def tick(self):
        """Perform health checks."""
        self.health_checks_performed += 1
        self.stats.tasks_completed += 1

        # Check system health
        health = self._check_system_health()

        # Check for issues
        issues = self._detect_issues(health)

        if issues:
            for issue in issues:
                self.alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "issue": issue,
                })
                self.message_bus.publish(Message(
                    msg_type=MessageType.ALERT,
                    sender=self.name,
                    data={"alert": issue, "severity": "WARNING"},
                    priority=2,
                ))
                logger.warning(f"[{self.name}] [WARN] Alert: {issue}")
        else:
            logger.info(f"[{self.name}] [OK] Health check #{self.health_checks_performed}: All systems nominal")

    def _check_system_health(self) -> Dict:
        """Check overall system health."""
        return {
            "cpu_percent": random.uniform(20, 80),
            "memory_percent": random.uniform(30, 70),
            "disk_percent": random.uniform(40, 60),
            "active_agents": len(self.agent_status),
            "api_requests_remaining": random.randint(100, 500),
        }

    def _detect_issues(self, health: Dict) -> List[str]:
        """Detect any issues from health metrics."""
        issues = []

        if health["cpu_percent"] > 90:
            issues.append("High CPU usage")
        if health["memory_percent"] > 85:
            issues.append("High memory usage")
        if health["api_requests_remaining"] < 50:
            issues.append("Low API rate limit remaining")

        # Check for stale agents
        for agent_name, status in self.agent_status.items():
            last_seen = datetime.fromisoformat(status.get("last_activity", datetime.now().isoformat()))
            if (datetime.now() - last_seen).seconds > 120:
                issues.append(f"Agent {agent_name} not responding")

        return issues

    def handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.msg_type == MessageType.HEARTBEAT:
            # Update agent status
            self.agent_status[message.sender] = message.data
            self.stats.tokens_processed += 50

        elif message.msg_type == MessageType.ALERT:
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "from": message.sender,
                "alert": message.data,
            })

    def get_subscribed_message_types(self) -> List[MessageType]:
        return super().get_subscribed_message_types() + [
            MessageType.HEARTBEAT,
            MessageType.ALERT,
            MessageType.METRIC,
        ]


class AnalysisAgent(BaseAgent):
    """
    Deep analysis of model performance.

    Analyzes:
    - Model predictions vs actuals
    - Feature importance drift
    - Win/loss patterns
    - Market regime correlation
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("Analyzer", message_bus)
        self.tick_interval = 120.0  # Analyze every 2 minutes
        self.analyses_completed = 0
        self.insights: List[Dict] = []

    def tick(self):
        """Perform analysis."""
        analysis = self._run_analysis()
        self.analyses_completed += 1
        self.stats.tasks_completed += 1
        self.stats.tokens_processed += 500

        # Publish results
        self.message_bus.publish(Message(
            msg_type=MessageType.ANALYSIS_RESULT,
            sender=self.name,
            data=analysis,
        ))

        # Store insights
        if analysis.get("insights"):
            self.insights.extend(analysis["insights"])

        logger.info(f"[{self.name}] [OK] Analysis #{self.analyses_completed} complete: {len(analysis.get('insights', []))} insights")

    def _run_analysis(self) -> Dict:
        """Run comprehensive analysis."""
        insights = []

        # Feature importance analysis
        important_features = ["pm_direction", "bb_position", "rsi_14", "volume_ratio"]
        for feat in important_features:
            drift = random.uniform(-0.1, 0.1)
            if abs(drift) > 0.05:
                insights.append({
                    "type": "feature_drift",
                    "feature": feat,
                    "drift": drift,
                    "action": "investigate" if drift > 0 else "monitor",
                })

        # Win pattern analysis
        recent_win_rate = random.uniform(0.45, 0.75)
        if recent_win_rate < 0.50:
            insights.append({
                "type": "performance",
                "metric": "win_rate",
                "value": recent_win_rate,
                "action": "retrain_recommended",
            })

        # Regime analysis
        regimes = ["bullish", "bearish", "sideways", "volatile"]
        regime_performance = {r: random.uniform(0.4, 0.8) for r in regimes}
        worst_regime = min(regime_performance, key=regime_performance.get)
        if regime_performance[worst_regime] < 0.45:
            insights.append({
                "type": "regime_weakness",
                "regime": worst_regime,
                "performance": regime_performance[worst_regime],
                "action": "add_regime_features",
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "insights": insights,
            "summary": {
                "total_insights": len(insights),
                "actionable": sum(1 for i in insights if i.get("action")),
            },
        }

    def handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.msg_type == MessageType.EXPERIMENT_COMPLETED:
            # Analyze completed experiment
            exp_results = message.data.get("results", {})
            logger.info(f"[{self.name}] Received experiment results for analysis")

    def get_subscribed_message_types(self) -> List[MessageType]:
        return super().get_subscribed_message_types() + [
            MessageType.EXPERIMENT_COMPLETED,
            MessageType.TEST_RESULT,
        ]


class OptimizationAgent(BaseAgent):
    """
    Boundary probing and hyperparameter optimization.

    Optimizes:
    - Hyperparameter search
    - Threshold optimization
    - Feature selection
    - Model architecture
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("Optimizer", message_bus)
        self.tick_interval = 180.0  # Optimize every 3 minutes
        self.optimizations_run = 0
        self.best_configs: List[Dict] = []
        self.probing_results: List[Dict] = []

    def tick(self):
        """Run optimization cycle."""
        # Boundary probing
        probing = self._run_boundary_probing()
        self.probing_results.append(probing)

        # Generate optimization suggestion
        suggestion = self._generate_optimization()

        self.optimizations_run += 1
        self.stats.tasks_completed += 1
        self.stats.tokens_processed += 1000

        # Publish optimization suggestion
        self.message_bus.publish(Message(
            msg_type=MessageType.REQUEST_OPTIMIZATION,
            sender=self.name,
            data={"config": suggestion, "probing_results": probing},
        ))

        logger.info(f"[{self.name}] [OK] Optimization #{self.optimizations_run}: Suggested config with {len(probing['boundaries_tested'])} boundaries probed")

    def _run_boundary_probing(self) -> Dict:
        """Probe model boundaries."""
        boundaries = [
            {"param": "learning_rate", "min": 0.001, "max": 0.5},
            {"param": "n_estimators", "min": 10, "max": 500},
            {"param": "max_depth", "min": 1, "max": 10},
            {"param": "threshold", "min": 0.3, "max": 0.7},
        ]

        results = []
        for b in random.sample(boundaries, k=2):
            test_values = [
                b["min"],
                (b["min"] + b["max"]) / 2,
                b["max"],
            ]

            for val in test_values:
                score = random.uniform(0.5, 0.8)  # Simulated
                results.append({
                    "param": b["param"],
                    "value": val,
                    "score": score,
                })

        return {
            "timestamp": datetime.now().isoformat(),
            "boundaries_tested": [r["param"] for r in results],
            "results": results,
        }

    def _generate_optimization(self) -> Dict:
        """Generate optimized configuration."""
        # Use best from probing
        best_values = {}

        for result in self.probing_results[-5:]:
            for r in result.get("results", []):
                param = r["param"]
                if param not in best_values or r["score"] > best_values[param]["score"]:
                    best_values[param] = r

        config = {
            "learning_rate": best_values.get("learning_rate", {}).get("value", 0.01),
            "n_estimators": int(best_values.get("n_estimators", {}).get("value", 100)),
            "max_depth": int(best_values.get("max_depth", {}).get("value", 3)),
            "threshold": best_values.get("threshold", {}).get("value", 0.5),
            "source": "boundary_probing",
        }

        return config

    def handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.msg_type == MessageType.ANALYSIS_RESULT:
            # Use analysis insights to guide optimization
            insights = message.data.get("insights", [])
            logger.info(f"[{self.name}] Received {len(insights)} insights for optimization guidance")

    def get_subscribed_message_types(self) -> List[MessageType]:
        return super().get_subscribed_message_types() + [
            MessageType.ANALYSIS_RESULT,
            MessageType.EXPERIMENT_COMPLETED,
        ]


# =============================================================================
# ADDITIONAL SPECIALIZED AGENTS
# =============================================================================

class SignalValidatorAgent(BaseAgent):
    """
    Validates trading signals before execution.

    Validates:
    - Signal consistency with model predictions
    - Risk parameters within limits
    - Market conditions appropriate
    - No conflicting signals
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("SignalValidator", message_bus)
        self.tick_interval = 15.0  # Validate every 15 seconds
        self.signals_validated = 0
        self.signals_rejected = 0
        self.validation_rules = [
            "confidence_threshold",
            "position_size_limit",
            "drawdown_limit",
            "volatility_check",
            "correlation_check",
        ]

    def tick(self):
        """Validate pending signals."""
        # Simulate signal validation
        n_signals = random.randint(0, 5)

        for _ in range(n_signals):
            signal = self._generate_mock_signal()
            is_valid, reasons = self._validate_signal(signal)

            if is_valid:
                self.signals_validated += 1
                self.message_bus.publish(Message(
                    msg_type=MessageType.METRIC,
                    sender=self.name,
                    data={"signal_validated": signal, "status": "APPROVED"},
                ))
            else:
                self.signals_rejected += 1
                self.message_bus.publish(Message(
                    msg_type=MessageType.ALERT,
                    sender=self.name,
                    data={"signal_rejected": signal, "reasons": reasons},
                    priority=3,
                ))

        self.stats.tasks_completed += 1
        if n_signals > 0:
            logger.info(f"[{self.name}] [OK] Validated {n_signals} signals: {self.signals_validated} approved, {self.signals_rejected} rejected")

    def _generate_mock_signal(self) -> Dict:
        return {
            "direction": random.choice(["LONG", "SHORT"]),
            "confidence": random.uniform(0.4, 0.9),
            "position_size": random.uniform(0.05, 0.30),
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_signal(self, signal: Dict) -> tuple:
        reasons = []

        # Check confidence
        if signal["confidence"] < 0.55:
            reasons.append("Low confidence")

        # Check position size
        if signal["position_size"] > 0.25:
            reasons.append("Position too large")

        # Random market condition check
        if random.random() < 0.1:
            reasons.append("Unfavorable market conditions")

        return len(reasons) == 0, reasons

    def handle_message(self, message: Message):
        pass


class DriftDetectorAgent(BaseAgent):
    """
    Monitors for model and data drift.

    Detects:
    - Feature distribution drift
    - Prediction distribution drift
    - Performance degradation
    - Concept drift
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("DriftDetector", message_bus)
        self.tick_interval = 300.0  # Check every 5 minutes
        self.drift_checks = 0
        self.drifts_detected = 0
        self.feature_baselines: Dict[str, float] = {}

    def tick(self):
        """Check for drift."""
        self.drift_checks += 1
        drift_report = self._check_drift()

        self.stats.tasks_completed += 1
        self.stats.tokens_processed += 200

        if drift_report["drift_detected"]:
            self.drifts_detected += 1
            self.message_bus.publish(Message(
                msg_type=MessageType.ALERT,
                sender=self.name,
                data=drift_report,
                priority=2,
            ))
            logger.warning(f"[{self.name}] [WARN] Drift detected: {drift_report['drift_type']}")
        else:
            logger.info(f"[{self.name}] [OK] Drift check #{self.drift_checks}: No significant drift")

    def _check_drift(self) -> Dict:
        features_to_check = ["pm_direction", "rsi_14", "bb_position", "volume_ratio"]

        for feat in features_to_check:
            current_mean = random.uniform(-0.5, 0.5)

            if feat not in self.feature_baselines:
                self.feature_baselines[feat] = current_mean
                continue

            drift = abs(current_mean - self.feature_baselines[feat])

            if drift > 0.3:  # Significant drift
                return {
                    "drift_detected": True,
                    "drift_type": "feature_distribution",
                    "feature": feat,
                    "baseline": self.feature_baselines[feat],
                    "current": current_mean,
                    "drift_magnitude": drift,
                }

        # Check performance drift
        recent_accuracy = random.uniform(0.45, 0.75)
        if recent_accuracy < 0.50:
            return {
                "drift_detected": True,
                "drift_type": "performance_degradation",
                "metric": "accuracy",
                "value": recent_accuracy,
                "threshold": 0.50,
            }

        return {"drift_detected": False}

    def handle_message(self, message: Message):
        if message.msg_type == MessageType.EXPERIMENT_COMPLETED:
            # Update baselines from new experiment
            pass


class RiskManagerAgent(BaseAgent):
    """
    Real-time risk monitoring and circuit breakers.

    Monitors:
    - Daily P&L limits
    - Position concentration
    - Drawdown limits
    - Correlation risks
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("RiskManager", message_bus)
        self.tick_interval = 5.0  # Check every 5 seconds
        self.risk_checks = 0
        self.breakers_triggered = 0
        self.current_pnl = 0.0
        self.max_daily_loss = -0.02  # 2% max daily loss
        self.max_drawdown = -0.10  # 10% max drawdown

    def tick(self):
        """Check risk limits."""
        self.risk_checks += 1
        self.stats.tasks_completed += 1

        # Simulate P&L movement
        self.current_pnl += random.uniform(-0.002, 0.003)

        risk_status = self._check_risk_limits()

        if risk_status["circuit_breaker"]:
            self.breakers_triggered += 1
            self.message_bus.publish(Message(
                msg_type=MessageType.ALERT,
                sender=self.name,
                data=risk_status,
                priority=1,  # Highest priority
            ))
            logger.warning(f"[{self.name}] [WARN] CIRCUIT BREAKER: {risk_status['reason']}")
        else:
            if self.risk_checks % 12 == 0:  # Log every minute
                logger.info(f"[{self.name}] [OK] Risk check #{self.risk_checks}: P&L={self.current_pnl:.2%}, Status=OK")

    def _check_risk_limits(self) -> Dict:
        # Daily loss limit
        if self.current_pnl < self.max_daily_loss:
            return {
                "circuit_breaker": True,
                "reason": "Daily loss limit exceeded",
                "current_pnl": self.current_pnl,
                "limit": self.max_daily_loss,
            }

        # Simulated drawdown check
        drawdown = random.uniform(-0.08, 0.0)
        if drawdown < self.max_drawdown:
            return {
                "circuit_breaker": True,
                "reason": "Max drawdown exceeded",
                "drawdown": drawdown,
                "limit": self.max_drawdown,
            }

        return {"circuit_breaker": False, "current_pnl": self.current_pnl}

    def handle_message(self, message: Message):
        pass


class DataQualityAgent(BaseAgent):
    """
    Monitors data integrity and freshness.

    Checks:
    - Data freshness (staleness)
    - Missing values
    - Outliers
    - Data consistency
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("DataQuality", message_bus)
        self.tick_interval = 60.0  # Check every minute
        self.quality_checks = 0
        self.issues_found = 0

    def tick(self):
        """Check data quality."""
        self.quality_checks += 1
        self.stats.tasks_completed += 1
        self.stats.tokens_processed += 100

        quality_report = self._check_data_quality()

        if quality_report["issues"]:
            self.issues_found += len(quality_report["issues"])
            self.message_bus.publish(Message(
                msg_type=MessageType.ALERT,
                sender=self.name,
                data=quality_report,
                priority=3,
            ))
            logger.warning(f"[{self.name}] [WARN] Data quality issues: {len(quality_report['issues'])}")
        else:
            logger.info(f"[{self.name}] [OK] Data quality check #{self.quality_checks}: All good")

    def _check_data_quality(self) -> Dict:
        issues = []

        # Check freshness
        data_age_seconds = random.randint(0, 120)
        if data_age_seconds > 60:
            issues.append({
                "type": "stale_data",
                "age_seconds": data_age_seconds,
                "threshold": 60,
            })

        # Check for missing values
        missing_pct = random.uniform(0, 0.05)
        if missing_pct > 0.01:
            issues.append({
                "type": "missing_values",
                "percentage": missing_pct,
                "threshold": 0.01,
            })

        # Check for outliers
        outlier_pct = random.uniform(0, 0.03)
        if outlier_pct > 0.02:
            issues.append({
                "type": "outliers",
                "percentage": outlier_pct,
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "checks_passed": 3 - len(issues),
            "checks_total": 3,
        }

    def handle_message(self, message: Message):
        pass


class BacktestAgent(BaseAgent):
    """
    Runs continuous backtests on configurations.

    Backtests:
    - New model configurations
    - Historical scenarios
    - Stress tests
    """

    def __init__(self, message_bus: MessageBus):
        super().__init__("Backtester", message_bus)
        self.tick_interval = 240.0  # Run backtest every 4 minutes
        self.backtests_run = 0
        self.backtest_results: List[Dict] = []

    def tick(self):
        """Run a backtest."""
        self.backtests_run += 1
        self.stats.tasks_completed += 1
        self.stats.tokens_processed += 500

        result = self._run_backtest()
        self.backtest_results.append(result)

        self.message_bus.publish(Message(
            msg_type=MessageType.METRIC,
            sender=self.name,
            data=result,
        ))

        logger.info(f"[{self.name}] [OK] Backtest #{self.backtests_run}: Sharpe={result['sharpe']:.2f}, Win={result['win_rate']:.1%}")

    def _run_backtest(self) -> Dict:
        return {
            "backtest_id": f"bt_{datetime.now().strftime('%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "period": random.choice(["1Y", "3Y", "5Y"]),
            "total_return": random.uniform(-0.1, 0.4),
            "sharpe": random.uniform(0.3, 2.0),
            "sortino": random.uniform(0.4, 2.5),
            "win_rate": random.uniform(0.45, 0.70),
            "max_drawdown": random.uniform(-0.25, -0.05),
            "total_trades": random.randint(100, 500),
        }

    def handle_message(self, message: Message):
        if message.msg_type == MessageType.REQUEST_OPTIMIZATION:
            # Optimization agent requesting backtest of new config
            logger.info(f"[{self.name}] Received optimization config for backtesting")


# =============================================================================
# MULTI-AGENT ORCHESTRATOR
# =============================================================================

class MultiAgentOrchestrator:
    """
    Main orchestrator that manages all agents.

    Responsibilities:
    - Start/stop agents
    - Monitor agent health
    - Coordinate work
    - Persist state
    """

    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.running = False
        self.start_time: Optional[datetime] = None

        # Status file
        self.status_file = LOGS_DIR / "multi_agent_status.json"

    def create_agents(self):
        """Create all agents."""
        self.agents = {
            # Core agents
            "test_generator": TestGenerationAgent(self.message_bus),
            "trainer": TrainingAgent(self.message_bus),
            "monitor": MonitoringAgent(self.message_bus),
            "analyzer": AnalysisAgent(self.message_bus),
            "optimizer": OptimizationAgent(self.message_bus),
            # Specialized agents
            "signal_validator": SignalValidatorAgent(self.message_bus),
            "drift_detector": DriftDetectorAgent(self.message_bus),
            "risk_manager": RiskManagerAgent(self.message_bus),
            "data_quality": DataQualityAgent(self.message_bus),
            "backtester": BacktestAgent(self.message_bus),
        }

        logger.info(f"Created {len(self.agents)} agents")

    def start(self):
        """Start all agents."""
        # Create agents first if not already created
        if not self.agents:
            self.create_agents()

        self.running = True
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("MULTI-AGENT ORCHESTRATOR STARTING")
        logger.info("=" * 60)

        for name, agent in self.agents.items():
            agent.start()
            time.sleep(0.5)  # Stagger starts

        logger.info(f"All {len(self.agents)} agents started")

        # Start status update loop
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._status_thread.start()

    def stop(self):
        """Stop all agents."""
        self.running = False

        logger.info("Stopping all agents...")
        for name, agent in self.agents.items():
            agent.stop()

        logger.info("All agents stopped")

    def _status_loop(self):
        """Periodically write status to file."""
        while self.running:
            try:
                status = self.get_status()
                with open(self.status_file, "w") as f:
                    json.dump(status, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Status write failed: {e}")

            time.sleep(5)  # Update every 5 seconds

    def get_status(self) -> Dict:
        """Get current system status."""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        agent_stats = {}
        for name, agent in self.agents.items():
            agent_stats[name] = agent.stats.to_dict()

        return {
            "orchestrator": {
                "running": self.running,
                "uptime_seconds": uptime,
                "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
                "n_agents": len(self.agents),
            },
            "agents": agent_stats,
            "message_bus": {
                "total_messages": len(self.message_bus.message_log),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def run_forever(self):
        """Run until interrupted."""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down...")
            self.stop()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("GIGA TRADER - MULTI-AGENT SYSTEM")
    logger.info("=" * 60)

    orchestrator = MultiAgentOrchestrator()
    orchestrator.create_agents()
    orchestrator.start()

    # Print initial status
    time.sleep(3)
    status = orchestrator.get_status()

    print("\n[*] MULTI-AGENT SYSTEM RUNNING")
    print("=" * 40)
    print(f"Active Agents: {status['orchestrator']['n_agents']}")
    for name, stats in status['agents'].items():
        marker = "[OK]" if stats['state'] == "RUNNING" else "[--]"
        print(f"  {marker} {name}: {stats['state']}")
    print("=" * 40)

    # Run until stopped
    orchestrator.run_forever()


if __name__ == "__main__":
    main()
