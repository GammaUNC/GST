#ifndef __CODEC_PIPELINE_H__
#define __CODEC_PIPELINE_H__

#include <memory>

namespace GenTC {

template<typename InType, typename OutType>
class PipelineUnit {
 public:
  typedef std::unique_ptr<OutType> ReturnType;
  typedef OutType ReturnValueType;
  PipelineUnit<InType, OutType>() { }
  virtual ~PipelineUnit<InType, OutType>() { }
  
  virtual ReturnType Run(const std::unique_ptr<InType> &in) const = 0;
};

template<typename InType, typename IntermediateType, typename OutType>
class PipelineChain : PipelineUnit<InType, OutType> {
 public:
  typedef PipelineUnit<InType, OutType> Base;
  PipelineChain(std::unique_ptr<PipelineUnit<InType, IntermediateType> > a,
                std::unique_ptr<PipelineUnit<IntermediateType, OutType> > b)
    : Base()
    , _first(std::move(a))
    , _second(std::move(b)) { }

  typename Base::ReturnType Run(const std::unique_ptr<InType> &in) const override {
    return std::move(_second->Run(_first->Run(in)));
  }

 private:
  std::unique_ptr<PipelineUnit<InType, IntermediateType> > _first;
  std::unique_ptr<PipelineUnit<IntermediateType, OutType> > _second;
};

template<typename InType, typename OutType>
class Pipeline {
 public:
  Pipeline<InType, OutType>(std::unique_ptr<PipelineUnit<InType, OutType> > &&unit)
    : _alg(std::move(unit)) { }

  template<typename NextType> std::unique_ptr<Pipeline<InType, NextType> >
    Chain(std::unique_ptr<PipelineUnit<OutType, NextType> > &&next) {
    typedef PipelineChain<InType, OutType, NextType> ChainType;
    typedef Pipeline<InType, NextType> OutputType;
    std::unique_ptr<ChainType> chain =
      std::unique_ptr<ChainType>(new ChainType(std::move(_alg), std::move(next)));
    return std::move(std::unique_ptr<OutputType>(new OutputType(chain)));
  }

  std::unique_ptr<OutType> Run(const std::unique_ptr<InType> &in) {
    return _alg->Run(in);
  }

 private:
  std::unique_ptr<PipelineUnit<InType, OutType> > _alg;
};

template<typename Type>
class Source : PipelineUnit<void, Type> {
  Source<Type>() { };
  virtual ~Source<Type>() { };

  virtual std::unique_ptr<Type> Get() const override = 0;

 private:
  std::unique_ptr<Type> Run(const std::unique_ptr<int> &in) const override {
    return Get();
  }
};

template<typename Type>
class Sink : PipelineUnit<Type, int> {
  Sink<Type>() { };
  virtual ~Sink<Type>() { };

  virtual void Finish(const std::unique_ptr<Type> &in) const override = 0;

 private:
  std::unique_ptr<int> Run(const std::unique_ptr<Type> &in) const override {
    Finish(in);
    return nullptr;
  }
};

}  // namespace GenTC

#endif  // __CODEC_PIPELINE_H__
