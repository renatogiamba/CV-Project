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
        self.conv1 = torch.nn.Conv2d(in_features,in_features*4, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_features*4)
        self.ps = torch.nn.PixelShuffle(upscale_factor=2)
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ps(out)
        out = self.relu(out)
        return out

class GeneratorRN(torch.nn.Module):
    """
    Pytorch class for handling the Srgan Generator
    """
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorRN, self).__init__()
        
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
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            # Second discriminator block:state size. (128) x 24 x 24
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            # Third discriminator block:state size. (256) x 12 x 12
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True),
            # Last Conv layer:state size. (512) x 6 x 6
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True)
        )
        
        # classifier. (512) x 6 x 6
        self.classifier = torch.nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.model(x)
        out = self.classifier(out)
        return out


class ResidualDenseBlock(torch.nn.Module):
    """
    Pytorch subclass for handling the ResidualDenseBlock
    """
    def __init__(self,in_features):
        super(ResidualDenseBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2*in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3*in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(4*in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(5*in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.lrelu = torch.nn.LeakyReLU()

    
    def forward(self,x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x,x1),1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class ResidualInResidualDenseBlock(torch.nn.Module):
    """
    Pytorch subclass for handling the ResidualInResidualDenseBlock part of Esrgan Generator
    """
    def __init__(self, filters):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.dense_blocks1 = ResidualDenseBlock(filters) 
        self.dense_blocks2 = ResidualDenseBlock(filters)  
        self.dense_blocks3 = ResidualDenseBlock(filters)

    def forward(self, x):
        out = self.dense_blocks1(x)
        out = self.dense_blocks2(out)
        out = self.dense_blocks3(out)
        return out * 0.2 + x

class UpscaleDenseBlock(torch.nn.Module):
    """
    Pytorch subclass for handling the upscale part of the Esrgan Generator
    """
    def __init__(self,in_features):
        super(UpscaleDenseBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_features, in_features * 4, kernel_size=3, stride=1, padding=1)
        self.lrelu = torch.nn.LeakyReLU()
        self.ps = torch.nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.ps(out)
        return out

class GeneratorRRDB(torch.nn.Module):
    """
    Pytorch class for handling the Esrgan Generator
    """
    def __init__(self, channels, filters=64, num_res_blocks=16):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = torch.nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)

        # Residual blocks
        self.res_blocks = torch.nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])

        # Second conv layer post residual blocks
        self.conv2 = torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        #Upsampling layers
        upsampling = []
        for _ in range(2):
            upsampling.append(UpscaleDenseBlock(filters))
        self.upsampling = torch.nn.Sequential(*upsampling)

        #Final output block
        self.conv5 = torch.nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.lrelu = torch.nn.LeakyReLU()
        self.conv6 = torch.nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv5(out)
        out = self.lrelu(out)
        out = self.conv6(out)
        return out

