

def get_2d_projection( batch_activations): 
    
    batch_activations = torch.from_numpy( batch_activations)
    
    b, c, h, w = batch_activations.shape
    
    #x = rearrange(batch_activations, "b c h w -> b (h w) c")
    x = batch_activations.reshape( b, c, h * w).permute( 0, 2, 1)
    
    x_mean = x.mean(1, keepdim=True)
    
    x = x - x_mean

    U, S, VT = torch.linalg.svd( x )
    
    #transpose 

    #V = rearrange(VT, 'a b c -> a c b')
    V = VT.permute( 0, 2, 1)
    V = V[ :, :, 0 : 1 ]

    projection = torch.bmm(x, V).squeeze( -1 )
    
    #projection = rearrange( projection, 'b (h w) -> b h w', h = h, w = w)
    projection = projection.reshape( b, h, w)
    
    return projection.detach().numpy( )
