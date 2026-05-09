#include <torch/extension.h>
#include <vector>

torch::Tensor reservoir_forward_cpp(
    torch::Tensor x, torch::Tensor W_in_pr, torch::Tensor W_in_al,
    torch::Tensor bias, torch::Tensor axial_codes, torch::Tensor surprise_thresholds,
    torch::Tensor W_eff, torch::Tensor iz_idx, torch::Tensor pr_idx, torch::Tensor al_idx,
    float reserve_input_scaling, float iz_self_loop_strength,
    int iz_reasoning_steps, float conflict_threshold
) {
    int B = x.size(0), T = x.size(1), n = W_in_pr.size(0);
    auto device = x.device();
    auto options = torch::TensorOptions().device(device);
    auto act = torch::zeros({B, n}, options);
    auto acts_buf = torch::zeros({B, T, n}, options);
    auto p_weight = torch::sigmoid((axial_codes.select(1, 0) - 0.5) * 10.0);
    auto a_weight = torch::sigmoid((axial_codes.select(1, 1) - 0.5) * 10.0);
    auto W_iz = W_eff.index_select(0, iz_idx).index_select(1, iz_idx);

    for (int t = 0; t < T; ++t) {
        auto pr_emb = x.select(1, t);
        torch::Tensor al_emb;
        if (t >= 3) { int start = std::max(0, t - 10); al_emb = x.slice(1, start, t).mean(1); }
        else { al_emb = pr_emb; }
        auto pr_drive = torch::matmul(pr_emb, W_in_pr.transpose(0, 1));
        auto al_drive = torch::matmul(al_emb, W_in_al.transpose(0, 1));
        auto sensory = (pr_drive * p_weight + al_drive * a_weight) * reserve_input_scaling;
        auto candidate = torch::tanh(sensory + bias);
        auto surprise = torch::abs(candidate - act);
        auto mask = surprise > surprise_thresholds.unsqueeze(0);
        act = torch::where(mask, candidate, act);
        act = act + 0.1 * torch::matmul(act, W_eff.transpose(0, 1));
        auto pr_mean = act.index_select(1, pr_idx).mean(1);
        auto al_mean = act.index_select(1, al_idx).mean(1);
        float conflict = torch::abs(pr_mean - al_mean).mean().item<float>();
        if (conflict > conflict_threshold) {
            auto iz_act = act.index_select(1, iz_idx).clone();
            for (int s = 0; s < iz_reasoning_steps; ++s) {
                iz_act = torch::tanh(iz_act + iz_self_loop_strength * torch::matmul(iz_act, W_iz.transpose(0, 1)));
            }
            act.index_put_({iz_idx}, iz_act);
        }
        acts_buf.select(1, t).copy_(act);
    }
    return acts_buf;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reservoir_forward_cpp, "Reservoir forward (C++)");
}
