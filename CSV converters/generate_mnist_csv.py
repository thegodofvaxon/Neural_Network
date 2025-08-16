def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)

    header = ",".join([f"pixel{i}" for i in range(28*28)]) + ",label\n"
    o.write(header)

    for i in range(n):
        label_byte = l.read(1)
        if not label_byte:
            print(f"Label file ended early at image {i}")
            break
        label = ord(label_byte)
        pixels = []
        for j in range(28*28):
            pixel_byte = f.read(1)
            if not pixel_byte:
                print(f"Image file ended early at image {i}, pixel {j}")
                break
            pixels.append(str(ord(pixel_byte)))
        o.write(",".join(pixels) + f",{label}\n")

    f.close()
    o.close()
    l.close()

convert(r"C:\Users\danie\Downloads\Projects\Data\Neural Network\Datasets\Mnist\t10k-images.idx3-ubyte",
        r"C:\Users\danie\Downloads\Projects\Data\Neural Network\Datasets\Mnist\t10k-labels.idx1-ubyte",
        "train.csv", 60000)
convert(r"C:\Users\danie\Downloads\Projects\Data\Neural Network\Datasets\Mnist\train-images.idx3-ubyte",
        r"C:\Users\danie\Downloads\Projects\Data\Neural Network\Datasets\Mnist\train-labels.idx1-ubyte",
        "test.csv", 10000)

