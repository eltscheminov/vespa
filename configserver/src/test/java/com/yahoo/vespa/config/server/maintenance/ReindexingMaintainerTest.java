// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.config.server.maintenance;

import com.yahoo.vespa.config.server.application.ApplicationReindexing;
import org.junit.Test;

import java.time.Instant;
import java.util.List;

import static com.yahoo.vespa.config.server.maintenance.ReindexingMaintainer.withReady;
import static org.junit.Assert.assertEquals;

/**
 * @author jonmv
 */
public class ReindexingMaintainerTest {

    @Test
    public void testReadyComputation() {
        ApplicationReindexing reindexing = ApplicationReindexing.empty()
                                                                .withReady(Instant.ofEpochMilli(1 << 20))
                                                                .withPending("one", "a", 10)
                                                                .withReady("two", "b", Instant.ofEpochMilli(2))
                                                                .withPending("two", "b", 20)
                                                                .withReady("two", Instant.ofEpochMilli(2 << 10))
                                                                .withReady("one", "a", Instant.ofEpochMilli(1))
                                                                .withReady("two", "c", Instant.ofEpochMilli(3));

        assertEquals(reindexing,
                     withReady(reindexing, List.of(), Instant.EPOCH));

        assertEquals(reindexing,
                     withReady(reindexing, List.of(19L), Instant.EPOCH));

        assertEquals(reindexing.withReady("two", "b", Instant.MAX),
                     withReady(reindexing, List.of(20L), Instant.MAX));
    }

}