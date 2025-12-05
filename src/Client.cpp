#include "perception_on_depth/Client.h"


WebSocketClient::WebSocketClient(nlohmann::json config, int pool)
{
    m_config = config;

    m_client.clear_access_channels(websocketpp::log::alevel::all);
    m_client.clear_error_channels(websocketpp::log::elevel::all);

    m_client.init_asio();

    m_status.store(false);
    m_client.set_open_handler([this](websocketpp::connection_hdl hdl)
                              { this->on_open(hdl); });
    m_client.set_fail_handler([this](websocketpp::connection_hdl hdl)
                              {
        auto con = m_client.get_con_from_hdl(hdl);
        std::cout << "Error: " << con->get_ec() << std::endl; });

    m_client.set_message_handler([this](websocketpp::connection_hdl hdl, websocketpp::config::asio_client::message_type::ptr msg)
                                 { this->on_message(hdl, msg); });
    m_client.set_close_handler([this](websocketpp::connection_hdl hdl)
                               { this->on_close(hdl); });

    m_client.start_perpetual();
}

WebSocketClient::~WebSocketClient()
{
    m_client.stop_perpetual();
    websocketpp::lib::error_code ec;
    m_client.close(m_handle, websocketpp::close::status::normal, "close", ec);
    if (m_thread->joinable())
    {
        m_thread->join();
    }
}

bool WebSocketClient::Connect(const std::string uri)
{
    websocketpp::lib::error_code ec;
    auto con = m_client.get_connection(uri, ec);
    if (ec)
    {
        RCLCPP_INFO_STREAM(rclcpp::get_logger("Client"), "Error: " << ec.message());
        return false;
    }
    m_handle = con->get_handle();
    m_client.connect(con);
    m_thread = websocketpp::lib::make_shared<websocketpp::lib::thread>(&Client::run, &m_client);
    return true;
}

void WebSocketClient::on_open(websocketpp::connection_hdl hdl)
{
    std::cout << "try to connect server" << std::endl;
    m_status.store(true);
    std::cout << "connect success!!!" << std::endl;
}

void WebSocketClient::on_message(websocketpp::connection_hdl hdl, websocketpp::config::asio_client::message_type::ptr msg)
{
    // std::cout<<"server send message: "<<msg->get_payload()<<std::endl;
}

void WebSocketClient::on_close(websocketpp::connection_hdl hdl)
{
    std::cout << "websocket connection closed" << std::endl;
}

void WebSocketClient::SendMsg(const std::string message)
{
    if (!m_status.load())
    {
        return;
    }
    m_client.send(m_handle, message, websocketpp::frame::opcode::text);
}

std::string JWTGenerator::generate(std::string req_id, std::string secret)
{
    nlohmann::json header = {
        {"alg", "HS256"},
        {"typ", "JWS"}};

    auto now = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();

    nlohmann::json payload = {
        {"aud", "koko robot"},
        {"exp", now + 3600},
        {"iss", "www.kokobots.com"},
        {"req_id", req_id},
        {"sub", "robot access token"}};

    std::string header_str = header.dump();
    std::string payload_str = payload.dump();

    std::string encoded_header = base64url_encode(reinterpret_cast<const unsigned char *>(header_str.data()), header_str.size());
    std::string encoded_payload = base64url_encode(reinterpret_cast<const unsigned char *>(payload_str.data()), payload_str.size());

    std::string signing_input = encoded_header + "." + encoded_payload;

    unsigned char hash[32];
    unsigned int hash_len;
    HMAC(EVP_sha256(),
         secret.c_str(), secret.length(),
         reinterpret_cast<const unsigned char *>(signing_input.c_str()), signing_input.length(),
         hash, &hash_len);

    std::string encoded_signature = base64url_encode(hash, hash_len);

    return signing_input + "." + encoded_signature;
}

std::string JWTGenerator::base64url_encode(const unsigned char *data, size_t len)
{
    BIO *b64 = BIO_new(BIO_f_base64());
    BIO *bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, data, len);
    BIO_flush(bio);

    char *encoded_data = nullptr;
    long length = BIO_get_mem_data(bio, &encoded_data);

    std::string result(encoded_data, length);

    std::string output;
    for (char c : result)
    {
        if (c == '+')
            output += '-';
        else if (c == '/')
            output += '_';
        else if (c != '=')
            output += c;
    }

    BIO_free_all(bio);
    return output;
}