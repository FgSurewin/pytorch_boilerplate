import matplotlib.pyplot as plt


class ImageUtils:
    @staticmethod
    def show_image(img, title=None):
        # unnormalize image
        img = img / 2 + 0.5
        img_np = img.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(img_np)
        if title is not None:
            plt.title(title)
        plt.show()


class ModelUtils:
    @staticmethod
    def history_tensor2item(history, round_digits=2):
        result = {}
        for key, value in history.items():
            result[key] = round(value.item(), round_digits)
        return result

    @staticmethod
    def get_model_acceleration(model):
        return next(model.parameters()).device
