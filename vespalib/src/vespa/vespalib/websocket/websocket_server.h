// Copyright 2016 Yahoo Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "handler.h"
#include "acceptor.h"
#include <vespa/vespalib/stllike/string.h>
#include <map>

namespace vespalib {
namespace ws {

class WebsocketServer : public Handler<Socket> {
public:
    struct StaticPage {
        StaticPage(StaticPage &&) = default;
        StaticPage & operator = (StaticPage &&) = default;
        ~StaticPage();
        vespalib::string content_type;
        vespalib::string content;
    };
    typedef std::map<vespalib::string, StaticPage> StaticRepo;

private:
    Acceptor         _acceptor;
    StaticRepo       _static_repo;
    vespalib::string _self;

public:
    WebsocketServer(int port_in, StaticRepo &&repo = StaticRepo());
    ~WebsocketServer();
    void handle(std::unique_ptr<Socket> socket) override;
    int port() { return _acceptor.port(); }
};

} // namespace vespalib::ws
} // namespace vespalib
