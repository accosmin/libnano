#include <mutex>
#include <nano/logger.h>
#include <nano/tensor/stream.h>
#include <nano/gboost/wlearner_dstep.h>
#include <nano/gboost/wlearner_dtree.h>
#include <nano/gboost/wlearner_hinge.h>
#include <nano/gboost/wlearner_stump.h>
#include <nano/gboost/wlearner_table.h>
#include <nano/gboost/wlearner_affine.h>

using namespace nano;

void wlearner_t::batch(int batch)
{
    m_batch = batch;
}

void wlearner_t::read(std::istream& stream)
{
    serializable_t::read(stream);

    int32_t ibatch = 0;

    critical(
        !::nano::read(stream, ibatch),
        "weak learner: failed to read from stream!");

    batch(ibatch);
}

void wlearner_t::write(std::ostream& stream) const
{
    serializable_t::write(stream);

    critical(
        !::nano::write(stream, static_cast<int32_t>(batch())),
        "weak learner: failed to write to stream!");
}

void wlearner_t::check(const indices_t& samples)
{
    critical(
        !std::is_sorted(::nano::begin(samples), ::nano::end(samples)),
        "weak learner: samples must be sorted by index!");
}

void wlearner_t::scale(tensor4d_t& tables, const vector_t& scale)
{
    critical(
        scale.size() != 1 && scale.size() != tables.size<0>(),
        "weak learner: mis-matching scale!");

    critical(
        scale.minCoeff() < 0,
        "weak learner: invalid scale factors!");

    for (tensor_size_t i = 0; i < tables.size<0>(); ++ i)
    {
        tables.array(i) *= scale(std::min(i, scale.size() - 1));
    }
}

tensor4d_t wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples) const
{
    tensor4d_t outputs(cat_dims(samples.size(), dataset.tdims()));
    outputs.zero();
    predict(dataset, samples, outputs);

    return outputs;
}

wlearner_factory_t& wlearner_t::all()
{
    static wlearner_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add_by_type<wlearner_lin1_t>();
        manager.add_by_type<wlearner_log1_t>();
        manager.add_by_type<wlearner_cos1_t>();
        manager.add_by_type<wlearner_sin1_t>();
        manager.add_by_type<wlearner_dstep_t>();
        manager.add_by_type<wlearner_dtree_t>();
        manager.add_by_type<wlearner_hinge_t>();
        manager.add_by_type<wlearner_stump_t>();
        manager.add_by_type<wlearner_table_t>();
    });

    return manager;
}
