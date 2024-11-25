from tinydb import TinyDB, Query
from datetime import datetime
import os

class RTLRAG:
    def __init__(self, db_path='db.json'):
        """Initialize database connection, creating path if needed"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        self.db = TinyDB(db_path)
        self.User = Query()
        print(f"Database initialized at: {os.path.abspath(db_path)}")

    def insert_one(self, data):
        """Insert a single document"""
        data['created_at'] = str(datetime.now())
        return self.db.insert(data)

    def insert_many(self, items):
        """Insert multiple documents"""
        for item in items:
            item['created_at'] = str(datetime.now())
        return self.db.insert_multiple(items)

    def find_one(self, query_dict):
        """Find matching document"""
        query = Query()
        conditions = [(getattr(query, k) == v) for k, v in query_dict.items()]
        return self.db.get(fold_criterion(conditions))

    def update(self, query_dict, new_data):
        """Update matching documents"""
        query = Query()
        conditions = [(getattr(query, k) == v) for k, v in query_dict.items()]
        new_data['updated_at'] = str(datetime.now())
        return self.db.update(new_data, fold_criterion(conditions))

    def delete(self, query_dict):
        """Delete matching documents"""
        query = Query()
        conditions = [(getattr(query, k) == v) for k, v in query_dict.items()]
        return self.db.remove(fold_criterion(conditions))

def fold_criterion(conditions):
    """Combine multiple conditions with AND logic"""
    if not conditions:
        return None
    result = conditions[0]
    for condition in conditions[1:]:
        result &= condition
    return result

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = RTLRAG('DB/example.json')
    
    # Insert documents
    db.insert_one({'name': 'John', 'age': 30, 'city': 'New York'})
    db.insert_many([
        {'name': 'Alice', 'age': 25, 'city': 'Boston'},
        {'name': 'Bob', 'age': 35, 'city': 'Chicago'}
    ])
    
    # Find documents
    result = db.find_one({'name': 'John'})
    
    print(result)
    # Update documents
    db.update({'name': 'John'}, {'age': 31})
    
    # Delete documents
    db.delete({'city': 'Chicago'})