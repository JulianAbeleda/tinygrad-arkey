import queue, threading

from tinygrad.helpers import db_connection


def test_diskcache_connection_is_thread_local():
  parent = db_connection()
  results = queue.Queue()
  def worker():
    connection = db_connection()
    results.put((connection is parent, connection.execute("SELECT 1").fetchone()))
  thread = threading.Thread(target=worker)
  thread.start()
  thread.join()
  same, row = results.get_nowait()
  assert not same
  assert row == (1,)
