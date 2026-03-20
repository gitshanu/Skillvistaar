import langchain, importlib.util
print('langchain', langchain.__version__)
print('schema', importlib.util.find_spec('langchain.schema'))
print('docstore', importlib.util.find_spec('langchain.docstore.document'))
try:
    from langchain.schema import Document
    print('schema Document ok', Document)
except Exception as e:
    print('schema Document failed', e)
try:
    from langchain.docstore.document import Document
    print('docstore Document ok', Document)
except Exception as e:
    print('docstore Document failed', e)
