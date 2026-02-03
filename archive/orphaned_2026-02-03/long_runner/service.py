"""
Windows Service Wrapper
=======================
Runs the orchestrator as a Windows service for 30+ day operation.

Features:
- Runs as Windows service
- Auto-start on boot
- Graceful shutdown handling
- Service recovery options
"""

import sys
import time
import logging
from pathlib import Path

# Try to import Windows-specific modules
try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("Warning: pywin32 not installed. Windows service features unavailable.")
    print("Install with: pip install pywin32")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("GigaTrader.Service")


if HAS_WIN32:
    class GigaTraderService(win32serviceutil.ServiceFramework):
        """
        Windows Service for GIGA TRADER Long-Running Orchestrator.

        Installation:
            python service.py install

        Start:
            python service.py start

        Stop:
            python service.py stop

        Remove:
            python service.py remove
        """

        _svc_name_ = "GigaTraderOrchestrator"
        _svc_display_name_ = "GIGA TRADER Experiment Orchestrator"
        _svc_description_ = (
            "Long-running automated trading experiment orchestrator. "
            "Runs continuous grid search optimization for 30+ days."
        )

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.orchestrator = None

        def SvcStop(self):
            """Handle service stop request."""
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)

            if self.orchestrator:
                logger.info("Service stop requested, stopping orchestrator...")
                self.orchestrator.stop()

        def SvcDoRun(self):
            """Main service entry point."""
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )

            self.main()

        def main(self):
            """Main service logic."""
            try:
                # Setup logging
                log_file = PROJECT_ROOT / "logs" / "service.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)

                handler = logging.FileHandler(log_file)
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s"
                ))
                logging.getLogger().addHandler(handler)
                logging.getLogger().setLevel(logging.INFO)

                logger.info("GIGA TRADER Service starting...")

                # Import and create orchestrator
                from src.long_runner.orchestrator import LongRunningOrchestrator, OrchestratorConfig

                config = OrchestratorConfig(
                    run_continuously=True,
                    target_experiments_per_day=50,
                )

                self.orchestrator = LongRunningOrchestrator(config=config)

                # Start orchestrator in a separate thread
                import threading
                orchestrator_thread = threading.Thread(
                    target=self.orchestrator.start,
                    daemon=True,
                )
                orchestrator_thread.start()

                # Wait for stop signal
                while True:
                    result = win32event.WaitForSingleObject(
                        self.stop_event,
                        5000,  # Check every 5 seconds
                    )

                    if result == win32event.WAIT_OBJECT_0:
                        logger.info("Stop signal received")
                        break

                    # Check if orchestrator is still running
                    if not orchestrator_thread.is_alive():
                        logger.error("Orchestrator thread died unexpectedly")
                        # Could restart here

                logger.info("Service stopped")

            except Exception as e:
                logger.error(f"Service error: {e}")
                servicemanager.LogErrorMsg(f"GIGA TRADER Service error: {e}")
                raise


def install_service():
    """Install the Windows service."""
    if not HAS_WIN32:
        print("Error: pywin32 required for Windows service")
        return False

    try:
        # Install the service
        win32serviceutil.InstallService(
            GigaTraderService._svc_name_,
            GigaTraderService._svc_display_name_,
            startType=win32service.SERVICE_AUTO_START,
            description=GigaTraderService._svc_description_,
        )
        print(f"Service '{GigaTraderService._svc_display_name_}' installed successfully")
        return True

    except Exception as e:
        print(f"Failed to install service: {e}")
        return False


def run_standalone():
    """Run orchestrator standalone (not as service)."""
    from src.long_runner.orchestrator import LongRunningOrchestrator, OrchestratorConfig

    print("=" * 60)
    print("GIGA TRADER - STANDALONE MODE")
    print("=" * 60)
    print("Running orchestrator in standalone mode...")
    print("Press Ctrl+C to stop")
    print()

    config = OrchestratorConfig(
        run_continuously=True,
        target_experiments_per_day=50,
    )

    orchestrator = LongRunningOrchestrator(config=config)

    try:
        orchestrator.start()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        orchestrator.stop()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GIGA TRADER Service Manager"
    )

    parser.add_argument(
        "action",
        choices=["install", "start", "stop", "restart", "remove", "standalone", "debug"],
        help="Service action",
    )

    args = parser.parse_args()

    if args.action == "standalone":
        run_standalone()
        return

    if not HAS_WIN32:
        print("Error: Windows service operations require pywin32")
        print("Install with: pip install pywin32")
        return

    if args.action == "install":
        win32serviceutil.HandleCommandLine(GigaTraderService, argv=["", "install"])
    elif args.action == "start":
        win32serviceutil.HandleCommandLine(GigaTraderService, argv=["", "start"])
    elif args.action == "stop":
        win32serviceutil.HandleCommandLine(GigaTraderService, argv=["", "stop"])
    elif args.action == "restart":
        win32serviceutil.HandleCommandLine(GigaTraderService, argv=["", "restart"])
    elif args.action == "remove":
        win32serviceutil.HandleCommandLine(GigaTraderService, argv=["", "remove"])
    elif args.action == "debug":
        win32serviceutil.HandleCommandLine(GigaTraderService, argv=["", "debug"])


if __name__ == "__main__":
    main()
