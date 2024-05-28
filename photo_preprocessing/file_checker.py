import os
def file_check(source):
    # Check if the file exists
    if not os.path.isfile(source):
        print('File not found: '+source)
        return False
    return True