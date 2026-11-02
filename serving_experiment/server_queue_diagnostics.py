"""Apply lightweight queue observation in the server and spawned engine."""

import os

from queue_observer import install
from vllm.v1.core.sched.request_queue import PrefixAwareRequestQueue

install(PrefixAwareRequestQueue, os.environ["CACHEWISE_QUEUE_TRACE"])

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
