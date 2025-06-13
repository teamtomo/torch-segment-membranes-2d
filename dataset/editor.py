from pathlib import Path

import napari
import mrcfile

import magicgui.widgets

viewer = napari.Viewer()

images = sorted(list(Path("images").glob("*/*.mrc")))
masks = sorted(list(Path("masks").glob("*/*.mrc")))

assert len(images) == len(masks)

global_idx = -1

def load_next_image():
    global global_idx
    global_idx += 1
    image_file = images[global_idx]
    mask_file = masks[global_idx]
    image = mrcfile.read(image_file)
    mask = mrcfile.read(mask_file)


    if "image" in viewer.layers:
        viewer.layers["image"].data = image
        viewer.layers["image"].metadata["path"] = image_file
        viewer.reset_view()
    else:
        viewer.add_image(image, name="image", metadata={"path": image_file})
        viewer.reset_view()
    if "mask" in viewer.layers:
        viewer.layers["mask"].data = mask
        viewer.layers["mask"].metadata["path"] = mask_file
    else:
        viewer.add_labels(mask, name="mask", metadata={"path": mask_file})


def save_segmentation():
    data = viewer.layers["mask"].data
    mask_path = viewer.layers["mask"].metadata["path"]
    mrcfile.write(mask_path, data, overwrite=True)

load_next_image_button = magicgui.widgets.PushButton(text='load next image')
load_next_image_button.clicked.connect(load_next_image)

save_button = magicgui.widgets.PushButton(text='save segmentation')
save_button.clicked.connect(save_segmentation)

widget = magicgui.widgets.Container()
widget.append(load_next_image_button)
widget.append(save_button)
viewer.window.add_dock_widget(widget)

napari.run()