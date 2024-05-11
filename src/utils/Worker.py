import threading

class Worker:
    _thread: threading.Thread = None

    def __init__(self, task: callable, args: tuple = ()):
        self._thread = threading.Thread(target=task, args=args)

    def start(self):
        return self._thread.start()
    
    def join(self):
        return self._thread.join()