class Entry:

    entries = []
    
    def __init__(self, **kwargs):
        self.ts = kwargs.get("ts")
        self.visitor_uuid = kwargs.get("visitor_uuid")
        self.visitor_source = kwargs.get("visitor_source")
        Entry.entries.append(self)

    def get_ts(self):
        return self.ts

    def get_visitor_uuid(self):
        return self.visitor_uuid
    
    def get_visitor_source(self):
        return self.visitor_source

pass