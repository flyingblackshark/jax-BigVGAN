import flax
import torch

def convert_torch_weights(path = "bigvgan_generator_3msteps.pt"):
    state_dict = torch.load(path,map_location=torch.device('cpu'))["generator"]
    params = {}
    # params["m_source.l_linear.kernel"] = state_dict["m_source.l_linear.weight"].T
    # params["m_source.l_linear.bias"] = state_dict["m_source.l_linear.bias"]
    params["conv_pre.layer_instance/kernel/scale"] = state_dict["conv_pre.weight_g"].squeeze((1, 2))
    params["conv_pre.layer_instance.kernel"] = state_dict["conv_pre.weight_v"].T
    params["conv_pre.layer_instance.bias"] = state_dict["conv_pre.bias"]
    # for i in range(5):
    #     params[f"noise_convs_{i}.kernel"] = state_dict[f"noise_convs.{i}.weight"].transpose(0,2)
    #     params[f"noise_convs_{i}.bias"] = state_dict[f"noise_convs.{i}.bias"]
    for i in range(6):
        params[f"ups_{i}.layer_instance/kernel/scale"] = state_dict[f"ups.{i}.0.weight_g"].squeeze((1, 2))
        params[f"ups_{i}.layer_instance.kernel"] = state_dict[f"ups.{i}.0.weight_v"].T
        params[f"ups_{i}.layer_instance.bias"] = state_dict[f"ups.{i}.0.bias"]
    for i in range(18):
        for j in range(1,3,1):
            for k in range(3):
                params[f"resblocks_{i}.convs{j}_{k}.layer_instance/kernel/scale"] = state_dict[f"resblocks.{i}.convs{j}.{k}.weight_g"].squeeze((1, 2))
                params[f"resblocks_{i}.convs{j}_{k}.layer_instance.kernel"] = state_dict[f"resblocks.{i}.convs{j}.{k}.weight_v"].T
                params[f"resblocks_{i}.convs{j}_{k}.layer_instance.bias"] = state_dict[f"resblocks.{i}.convs{j}.{k}.bias"]
                params[f"resblocks_{i}.activations_{(j-1)*3+k}.activation.alpha"] = state_dict[f"resblocks.{i}.activations.{(j-1)*3+k}.act.alpha"]
                params[f"resblocks_{i}.activations_{(j-1)*3+k}.activation.beta"] = state_dict[f"resblocks.{i}.activations.{(j-1)*3+k}.act.beta"]
    params[f"activation_post.activation.alpha"] = state_dict[f"activation_post.act.alpha"]
    params[f"activation_post.activation.beta"] = state_dict[f"activation_post.act.beta"]
    params["conv_post.layer_instance/kernel/scale"] = state_dict["conv_post.weight_g"].squeeze((1, 2))
    params["conv_post.layer_instance.kernel"] = state_dict["conv_post.weight_v"].T
    #params["conv_post.layer_instance.bias"] = state_dict["conv_post.bias"]
    params = {k: v.cpu().numpy() for k, v in params.items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    return params