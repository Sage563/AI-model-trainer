import api as modeltrian
modeltrian.main("samples/texts/libary.xml", None, "text_model.h5", 1)
#image is modeltrian.main(None, "path/to/image_folder", "image_model.h5", 3)


print("Model created")
