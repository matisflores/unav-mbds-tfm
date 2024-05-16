import queue
import threading

class EventListener:
    _thread: threading.Thread = None
    _events: queue.Queue = None
    _on_event: callable = None

    def __init__(self, on_event: callable, events: queue.Queue):
        self._on_event = on_event
        self._events = events

    def listen_events(self, events: queue.Queue):
        while True:
            event = events.get()

            if event is None:
                break

            self._on_event(event)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self.listen_events, args=(self._events,))
            self._thread.start()
    
    def stop(self):
        if self._thread is not None:
            self._events.put(None)
            self._thread.join()