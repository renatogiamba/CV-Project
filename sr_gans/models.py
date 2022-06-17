import torch

class ResidualBlock(torch.nn.Module):
    """
    Pytorch subclass for handling the Generator fundamental part: Residual Block
    """
    def __init__(self,in_features):
        super(ResidualBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_features, 0.8)
        self.relu = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(in_features, 0.8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out+x

class UpscaleBlock(torch.nn.Module):
    """
    Pytorch subclass for handling the upscale part of Generator
    """
    def __init__(self,in_features):
        super(UpscaleBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_features,in_features*4, kernel_size=3, stride=1, padding=1),
        self.bn1 = torch.nn.BatchNorm2d(in_features*4),
        self.ps = torch.nn.PixelShuffle(upscale_factor=2),
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ps(out)
        out = self.relu(out)
        return out

class Generator(torch.nn.Module):
    """
    Pytorch class for handling the Generator
    """
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()
        
        #First conv layer
        self.conv1 = torch.nn.Conv2d(in_channels,64,kernel_size=9,stride=1,padding=4)
        self.relu = torch.nn.PReLU()
    
        # Features residual block
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = torch.nn.Sequential(*res_blocks)
        
        #Second conv layer
        self.conv2 = torch.nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64, 0.8)
        
        # Upscale block
        upsampling = []
        for _ in range(2):
            upsampling.append(UpscaleBlock(64))
        self.upsampling = torch.nn.Sequential(*upsampling)
        
        #Output layer
        self.conv3 = torch.nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out2 = self.bn1(out2)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out

class Discriminator(torch.nn.Module):
    """
    Pytorch class for handling the Discriminator
    """
    def __init__(self,in_channels=3):
        super(Discriminator,self).__init__()
        self.model = torch.nn.Sequential(
            # First Conv layer:input size. (3) x 96 x 96
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.LeakyReLU(0.2, True),
            # First discriminator block:state size. (64) x 48 x 48
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=2, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            # Second discriminator block:state size. (128) x 24 x 24
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=2, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            # Third discriminator block:state size. (256) x 12 x 12
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=2, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True),
            # Last Conv layer:state size. (512) x 6 x 6
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True)
        )
        
        #classifier. (512) x 6 x 6
        self.classifier=torch.nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)


    def forward(self,x):
      out=self.model(x)
      out=self.classifier(out)
      return out
