class LRUCache:
    """
    A simple LRU cache implementation
    """
    def __init__(self, capacity=10000):
        self.cache = {}
        self.order = []
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]  
    
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) == self.capacity:
            self.cache.pop(self.order.pop(0))
        self.cache[key] = value
        self.order.append(key) 