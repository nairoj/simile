from utils import PSGenerator, Glove
if __name__=='__main__':
    generator = PSGenerator('/content/drive/MyDrive/simile-triple-completion/output/pretained_e1','/content/drive/MyDrive/simile-triple-completion/pretrained_models/glove/model.bin')
    print(generator.SI('love','sun'))
    print(generator.SG('silent','city'))