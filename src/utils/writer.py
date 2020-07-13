class Writer:

    def __init__(self,output_file):
        self.outfile = output_file


    def write(self,x,y):
        with open(self.outfile,'a') as f:
            f.write('{},{}'.format(x,y))
