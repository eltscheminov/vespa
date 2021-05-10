// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.controller.api.integration.certificates;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * This class is used for metadata about an application's endpoint certificate on the controller.
 * <p>
 * It has more properties than com.yahoo.config.model.api.EndpointCertificateMetadata.
 *
 * @author andreer
 */
public class EndpointCertificateMetadata {

    private final String keyName;
    private final String certName;
    private final int version;
    private final long lastRequested;
    private final String requestId;
    private final List<String> requestedDnsSans;
    private final String issuer;
    private final Optional<Long> expiry;
    private final Optional<Long> lastRefreshed;

    public EndpointCertificateMetadata(String keyName, String certName, int version, long lastRequested, String requestId, List<String> requestedDnsSans, String issuer, Optional<Long> expiry, Optional<Long> lastRefreshed) {
        this.keyName = keyName;
        this.certName = certName;
        this.version = version;
        this.lastRequested = lastRequested;
        this.requestId = requestId;
        this.requestedDnsSans = requestedDnsSans;
        this.issuer = issuer;
        this.expiry = expiry;
        this.lastRefreshed = lastRefreshed;
    }

    public String keyName() {
        return keyName;
    }

    public String certName() {
        return certName;
    }

    public int version() {
        return version;
    }

    public long lastRequested() {
        return lastRequested;
    }

    public String requestId() {
        return requestId;
    }

    public List<String> requestedDnsSans() {
        return requestedDnsSans;
    }

    public String issuer() {
        return issuer;
    }

    public Optional<Long> expiry() {
        return expiry;
    }

    public Optional<Long> lastRefreshed() {
        return lastRefreshed;
    }

    public EndpointCertificateMetadata withVersion(int version) {
        return new EndpointCertificateMetadata(
                this.keyName,
                this.certName,
                version,
                this.lastRequested,
                this.requestId,
                this.requestedDnsSans,
                this.issuer,
                this.expiry,
                this.lastRefreshed);
    }

    public EndpointCertificateMetadata withLastRequested(long lastRequested) {
        return new EndpointCertificateMetadata(
                this.keyName,
                this.certName,
                this.version,
                lastRequested,
                this.requestId,
                this.requestedDnsSans,
                this.issuer,
                this.expiry,
                this.lastRefreshed);
    }

    public EndpointCertificateMetadata withLastRefreshed(long lastRefreshed) {
        return new EndpointCertificateMetadata(
                this.keyName,
                this.certName,
                this.version,
                this.lastRequested,
                this.requestId,
                this.requestedDnsSans,
                this.issuer,
                this.expiry,
                Optional.of(lastRefreshed));
    }

    public EndpointCertificateMetadata withRequestId(String requestId) {
        return new EndpointCertificateMetadata(
                this.keyName,
                this.certName,
                this.version,
                this.lastRequested,
                requestId,
                this.requestedDnsSans,
                this.issuer,
                this.expiry,
                lastRefreshed);
    }

    @Override
    public String toString() {
        return "EndpointCertificateMetadata{" +
               "keyName='" + keyName + '\'' +
               ", certName='" + certName + '\'' +
               ", version=" + version +
               ", lastRequested=" + lastRequested +
               ", requestId=" + requestId +
               ", requestedDnsSans=" + requestedDnsSans +
               ", issuer=" + issuer +
               ", expiry=" + expiry +
               ", lastRefreshed=" + lastRefreshed +
               '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        EndpointCertificateMetadata that = (EndpointCertificateMetadata) o;
        return version == that.version &&
                lastRequested == that.lastRequested &&
               keyName.equals(that.keyName) &&
               certName.equals(that.certName) &&
               requestId.equals(that.requestId) &&
               requestedDnsSans.equals(that.requestedDnsSans) &&
               issuer.equals(that.issuer) &&
               expiry.equals(that.expiry) &&
               lastRefreshed.equals(that.lastRefreshed);
    }

    @Override
    public int hashCode() {
        return Objects.hash(keyName, certName, version, lastRequested, requestId, requestedDnsSans, issuer, expiry, lastRefreshed);
    }

}
