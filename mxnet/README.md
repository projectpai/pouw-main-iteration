Default MXNET package doesn't work on processors without some instructions support(like AVX2).
This Dockerfile allows to compile custom `wheel` package.
```
build.sh <version>
```
where `version` is relevant MXNET version. 
