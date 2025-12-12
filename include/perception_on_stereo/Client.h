#pragma once

#include "perception_on_stereo/common.h"

#include "websocketpp/client.hpp"
#include "websocketpp/config/asio_no_tls_client.hpp"

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using Client = websocketpp::client<websocketpp::config::asio_client>;

class WebSocketClient
{
public:
    WebSocketClient(const std::string uri): uri(uri) {
        m_client.clear_access_channels(websocketpp::log::alevel::all);
        m_client.clear_error_channels(websocketpp::log::elevel::all);

        m_client.init_asio();
        m_client.start_perpetual();

        m_client.set_open_handler(
            [this](websocketpp::connection_hdl hdl){
                this->on_open(hdl); 
            }
        );
        m_client.set_fail_handler(
            [this](websocketpp::connection_hdl hdl){
                this->on_fail(hdl); 
            }
        );
        m_client.set_message_handler(
            [this](websocketpp::connection_hdl hdl, websocketpp::config::asio_client::message_type::ptr msg){
                this->on_message(hdl, msg); 
            }
        );
        m_client.set_close_handler(
            [this](websocketpp::connection_hdl hdl){ 
                this->on_close(hdl); 
            }
        );

        m_thread = websocketpp::lib::make_shared<websocketpp::lib::thread>(&Client::run, &m_client);

        m_reconnect_thread_ = std::thread(
            [this](){
                std::this_thread::sleep_for(std::chrono::seconds(3));
                while(rclcpp::ok() && !reconnecting.load()){
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    if (connected.load()){
                        continue;
                    }
                    reconnecting.store(true);
                    std::string info = fmt::format("\33[33mReconnecting to {}...\33[0m", this->uri);
                    ROS_LOG(info.c_str());
                    connect();
                    reconnecting.store(false);
                }
            }
        );
        m_reconnect_thread_.detach();
    }
    ~WebSocketClient(){
        m_client.stop_perpetual();

        websocketpp::lib::error_code ec;
        m_client.close(m_handle, websocketpp::close::status::normal, "close", ec);

        if (m_thread->joinable())
        {
            m_thread->join();
        }
        if (m_reconnect_thread_.joinable())
        {
            m_reconnect_thread_.join();
        }
    }
public:
    void on_open(websocketpp::connection_hdl hdl){
        std::string info = fmt::format("\33[32mConnection uri {} opened\33[0m", uri);
        ROS_LOG(info.c_str());
        // RCLCPP_INFO_STREAM(rclcpp::get_logger("WebSocketClient"), info.c_str());
        connected.store(true);
    }
    void on_fail(websocketpp::connection_hdl hdl){
        auto con = m_client.get_con_from_hdl(hdl);
        std::cout << "Error: " << con->get_ec() << std::endl; 
    }
    void on_message(websocketpp::connection_hdl hdl, Client::message_ptr msg){
        std::string message = msg->get_payload();
        nlohmann::json data = nlohmann::json::parse(message);
    }
    void on_close(websocketpp::connection_hdl hdl){
        std::string info = fmt::format("\33[31mConnection uri {} closed\33[0m", uri);
        ROS_LOG(info.c_str());
        // RCLCPP_INFO_STREAM(rclcpp::get_logger("WebSocketClient"), info.c_str());
        connected.store(false);
    }
public:
    void connect(){
        websocketpp::lib::error_code ec;
        auto con = m_client.get_connection(uri, ec);
        if (ec)
        {
            ROS_LOG("Error: %s", ec.message().c_str());
            return;
            // RCLCPP_INFO_STREAM(rclcpp::get_logger("WebSocketClient"), "Error: " << ec.message());
        }
        m_handle = con->get_handle();
        m_client.connect(con);
    }
    void send_message(const std::string message){
        m_client.send(m_handle, message, websocketpp::frame::opcode::text);
    }
public:
    std::string uri;
    std::atomic<bool> connected{false};
    std::atomic<bool> reconnecting{false};
    std::atomic<bool> start_collect{false};
private:
    std::thread m_reconnect_thread_;
    Client m_client;
    websocketpp::connection_hdl m_handle;
    websocketpp::lib::shared_ptr<websocketpp::lib::thread> m_thread;
};

class UDPClient
{
public:
    UDPClient(const std::string ip, uint16_t port):sockfd(-1){
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);

        std::memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        server_addr.sin_addr.s_addr = inet_addr(ip.c_str());
    }
    ~UDPClient() {
        if (sockfd >= 0) {
            close(sockfd);
        }
    }

public:
    void SendMsg(const std::string message){
        if(sockfd >= 0){
            sendto(sockfd, message.c_str(), message.size(), 0, (struct sockaddr*)&server_addr, sizeof(server_addr));
        }
    }

public:
    int sockfd;
    struct sockaddr_in server_addr;
};
