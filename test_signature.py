from rag_engine import RAGEngine

def test_callback(msg):
    print(f"Status: {msg}")

try:
    engine = RAGEngine()
    # Dummy query to see if method signature matches
    print("Testing query method signature...")
    # Using a simple query that shouldn't trigger too much if possible, 
    # but we just want to see if the call itself fails with TypeError
    # We'll mock the internal calls or just let it fail naturally on something else 
    # as long as it's NOT a TypeError on the signature.
    engine.query("test", status_callback=test_callback)
    print("Success: Method signature accepted status_callback")
except TypeError as e:
    print(f"Failure: {e}")
except Exception as e:
    # Other exceptions are fine, we just care about the TypeError for the argument
    print(f"Method signature likely OK, failed later: {type(e).__name__}")
