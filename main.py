"""
The algorithm takes three images, an input image, a content-image, and a 
style-image, and changes the input to resemble the content of the content-image 
and the artistic style of the style-image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--content", "-c", help="content image", required=True)
    parser.add_argument("--style", "-s", help="style image", required=True)
    parser.add_argument("--out", "-o", help="path to save output image", required=True)
    parser.add_argument("--device", "-d", default="cuda:1", help="device")
    return parser.parse_args()


class ContentLoss(nn.Module):
    def __init__(self, target_feature) -> None:
        super().__init__()
        self.target = target_feature.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


def gram_matrix(input):
    n, c, h, w = input.size()  # n = batch size (=1)
    features = input.view(n * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(n * c * h * w)


class Normalization(nn.Module):
    def __init__(self, mean, std) -> None:
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    imsize = [512, 512] if torch.cuda.is_available() else [128, 128]

    transform = T.Compose([T.Resize(imsize), T.ToTensor()])

    def image_loader(image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = transform(image).unsqueeze(0)
        return image.to(device, torch.float)[:, :3, :, :]

    unloader = T.ToPILImage()

    content_img = image_loader(args.content)
    input_img = content_img.clone()
    style_img = image_loader(args.style)

    assert (
        style_img.size() == content_img.size()
    ), "we need to import style and content images of the same size, got {} and {}".format(
        style_img.size(), content_img.size()
    )

    cnn = (
        models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        .features.to(device)
        .eval()
    )
    # print(cnn)

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers_default = ["conv2_2"]
    style_layers_default = [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
    ]

    def get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        style_img,
        content_img,
        content_layers=content_layers_default,
        style_layers=style_layers_default,
    ):
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # just in order to have an iterable access to or list of content/style
        # losses
        content_losses = []
        style_losses = []

        # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i, j = 1, 1  # increment j every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                name = "conv{}_{}".format(i, j)
            elif isinstance(layer, nn.ReLU):
                name = "relu{}_{}".format(i, j)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
                j += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)
                i += 1
                j = 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn_{}".format(i)
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss{}_{}".format(i, j), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss{}_{}".format(i, j), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[: (i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(
        cnn,
        normalization_mean,
        normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=300,
        style_weight=1000000,
        content_weight=1,
    ):
        """Run the style transfer."""
        print("Building the style transfer model..")
        model, style_losses, content_losses = get_style_model_and_losses(
            cnn, normalization_mean, normalization_std, style_img, content_img
        )

        print(model)

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = get_input_optimizer(input_img)

        print("Optimizing..")
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print(
                        "Style Loss : {:4f} Content Loss: {:4f}".format(
                            style_score.item(), content_score.item()
                        )
                    )
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
    )

    output_img = unloader(output.squeeze(0))
    output_img.save(args.out)


if __name__ == "__main__":
    args = parse_args()
    main(args)
