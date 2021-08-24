#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <grpcpp/grpcpp.h>
#include "verifier.grpc.pb.h"
#include "task_info.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using pai::pouw::verification::Request;
using pai::pouw::verification::Response;
using pai::pouw::verification::Verifier;
using pai::pouw::verification::Response_ReturnCode;
using pai::pouw::verification::Response_ReturnCode_GENERAL_ERROR;

using pai::pouw::task_info::TaskListRequest;
using pai::pouw::task_info::TaskListResponse;
using pai::pouw::task_info::TaskInfo;


class VerificationClient {
 public:
  VerificationClient(std::shared_ptr<Channel> channel)
      : stub_(Verifier::NewStub(channel)) {}

  std::pair<Response_ReturnCode, std::string> TestServer() {

    Request request;
    request.set_msg_history_id("100");
    request.set_msg_id("100");
    request.set_nonce(1);

    Response response;
    ClientContext context;

    // The actual RPC.
    Status status = stub_->Verify(&context, request, &response);

    // Act upon its status.
    if (status.ok()) {
      return std::make_pair(response.code(), response.description());
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return std::make_pair(Response_ReturnCode_GENERAL_ERROR, "RPC failed");
    }
  }

 private:
  std::unique_ptr<Verifier::Stub> stub_;
};


class TaskListClient {
public:
    TaskListClient(std::shared_ptr<Channel> channel)
    : stub_(TaskInfo::NewStub(channel)) {}

    std::vector<std::string> TestGetWaitingTasks() {

        TaskListRequest request;

        TaskListResponse response;
        ClientContext context;

        // The actual RPC.
        Status status = stub_->GetWaitingTasks(&context, request, &response);

        // Act upon its status.
        if (status.ok()) {
            auto tasks = response.tasks();
            return std::vector<std::string>(tasks.begin(), tasks.end());
        } else {
            std::cout << status.error_code() << ": " << status.error_message()
            << std::endl;
            return std::vector<std::string>();
        }
    }

private:
    std::unique_ptr<TaskInfo::Stub> stub_;
};

int main(int argc, char** argv) {
  /*VerificationClient greeter(grpc::CreateChannel(
      "localhost:50011", grpc::InsecureChannelCredentials()));
  auto verificationResult = greeter.TestServer();
  std::cout << "Code:" << int(verificationResult.first) << std::endl;
  std::cout << "Description: " << verificationResult.second << std::endl;*/

  TaskListClient taskLister(grpc::CreateChannel(
          "localhost:50011", grpc::InsecureChannelCredentials()));
  auto taskList = taskLister.TestGetWaitingTasks();
  std::cout << "Tasks:\n";
  for (auto t : taskList)
      std::cout << t << "\n";

  return 0;
}
