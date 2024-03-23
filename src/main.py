from dataset_painter import SeedlingDataset


INPUT_FOLDER = './plantulas_soja/1'
OUTPUT_FOLDER = './dataset/plantulas_soja/1'

if __name__ == '__main__':
    sd = SeedlingDataset(INPUT_FOLDER, OUTPUT_FOLDER)
    sd.run()
