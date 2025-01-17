// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.feed.client;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

import static java.util.Objects.requireNonNull;

/**
 * @author jonmv
 */
public class DocumentId {

    private final String documentType;
    private final String namespace;
    private final OptionalLong number;
    private final Optional<String> group;
    private final String userSpecific;

    private DocumentId(String documentType, String namespace, OptionalLong number, Optional<String> group, String userSpecific) {
        this.documentType = requireNonNull(documentType);
        this.namespace = requireNonNull(namespace);
        this.number = requireNonNull(number);
        this.group = requireNonNull(group);
        this.userSpecific = requireNonNull(userSpecific);
    }

    public static DocumentId of(String namespace, String documentType, String userSpecific) {
        return new DocumentId(namespace, documentType, OptionalLong.empty(), Optional.empty(), userSpecific);
    }

    public static DocumentId of(String namespace, String documentType, long number, String userSpecific) {
        return new DocumentId(namespace, documentType, OptionalLong.of(number), Optional.empty(), userSpecific);
    }

    public static DocumentId of(String namespace, String documentType, String group, String userSpecific) {
        return new DocumentId(namespace, documentType, OptionalLong.empty(), Optional.of(group), userSpecific);
    }

    public static DocumentId of(String serialized) {
        String[] parts = serialized.split(":");
        while (parts.length >= 5 && parts[0].equals("id")) {
            if (parts[3].startsWith("n="))
                return DocumentId.of(parts[1], parts[2], Long.parseLong(parts[3]), parts[4]);
            if (parts[3].startsWith("g="))
                return DocumentId.of(parts[1], parts[2], parts[3], parts[4]);
            else if (parts[3].isEmpty())
                return DocumentId.of(parts[1], parts[2], parts[4]);
        }
        throw new IllegalArgumentException("Document ID must be on the form " +
                                           "'id:<namespace>:<document-type>:[n=number|g=group]:<user-specific>', " +
                                           "but was '" + serialized + "'");
    }

    public String documentType() {
        return documentType;
    }

    public String namespace() {
        return namespace;
    }

    public OptionalLong number() {
        return number;
    }

    public Optional<String> group() {
        return group;
    }

    public String userSpecific() {
        return userSpecific;
    }

}
