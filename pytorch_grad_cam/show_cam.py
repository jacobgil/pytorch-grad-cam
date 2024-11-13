import cv2
import numpy as np
import matplotlib.pyplot as plt
def show_img(cam, img):
#         cam_=cv2.resize(cam, (224, 224))
        heatmap = cv2.resize(cam.cpu().detach().numpy(), (img.shape[1], img.shape[0]))
#         print(heatmap)
#         print(cam)
        heatmap = np.uint8(255 * heatmap)
#         plt.imshow(heatmap)
#         plt.colorbar()
#         plt.show()
#         print(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
 

        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()