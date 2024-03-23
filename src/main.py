from dataset_painter import SeedlingDataset


input_folder = './plantulas_soja/1'
output_folder = './dataset/plantulas_soja/1'

if __name__ == '__main__':
    sd = SeedlingDataset(input_folder, output_folder)
    sd.run()
