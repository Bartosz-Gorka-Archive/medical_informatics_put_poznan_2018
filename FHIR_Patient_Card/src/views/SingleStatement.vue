<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Statement {{ get(['id'], selectedStatement)}}</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">
              <router-link :to="{ name: 'statements' }" class="c-breadcrumb__link">Medication Statements</router-link>
            </li>
            <li class="c-breadcrumb__item">Statement {{ get(['id'], selectedStatement)}}</li>
          </ol>
        </div>
      </div>
    </header>

    <table class="table table--data">
      <thead>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </thead>
      <tfoot>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </tfoot>
      <tbody>
        <tr v-if="get(['id'], selectedStatement)">
          <td data-label="Key">ID</td>
          <td data-label="Value">{{ get(['id'], selectedStatement) }}</td>
        </tr>

        <tr v-if="get(['status'], selectedStatement)">
          <td data-label="Key">Status</td>
          <td data-label="Value">{{ get(['status'], selectedStatement) }}</td>
        </tr>

        <tr v-if="get(['meta', 'versionId'], selectedStatement)">
          <td data-label="Key">Version ID</td>
          <td data-label="Value">{{ get(['meta', 'versionId'], selectedStatement) }}</td>
        </tr>

        <tr v-if="get(['meta', 'lastUpdated'], selectedStatement)">
          <td data-label="Key">Last updated</td>
          <td data-label="Value">{{ new Date(get(['meta', 'lastUpdated'], selectedStatement)).toLocaleString() }}</td>
        </tr>

        <tr v-if="get(['dosage', 0, 'text'], selectedStatement)">
          <td data-label="Key">Dosage</td>
          <td data-label="Value">{{ get(['dosage', 0, 'text'], selectedStatement) }}</td>
        </tr>

        <tr v-if="get(['medicationCodeableConcept', 'text'], selectedStatement)">
          <td data-label="Key">Concept</td>
          <td data-label="Value">{{ get(['medicationCodeableConcept', 'text'], selectedStatement) }}</td>
        </tr>

        <tr v-if="get(['medicationCodeableConcept', 'coding', 0, 'display'], selectedStatement)">
          <td data-label="Key">Concept</td>
          <td data-label="Value">{{ get(['medicationCodeableConcept', 'coding', 0, 'display'], selectedStatement) }}</td>
        </tr>

        <tr v-if="get(['reasonNotTaken', 0, 'coding', 0, 'display'], selectedStatement)">
          <td data-label="Key">Not taken - reason</td>
          <td data-label="Value">{{ get(['reasonNotTaken', 0, 'coding', 0, 'display'], selectedStatement) }}</td>
        </tr>

        <tr v-if="get(['taken'], selectedStatement)">
          <td data-label="Key">Taken</td>
          <td data-label="Value">{{ taken(get(['taken'], selectedStatement)) }}</td>
        </tr>

        <tr v-if="get(['subject', 'reference'], selectedStatement)">
          <td data-label="Key">Subject</td>
          <td data-label="Value">
            <router-link :to="{ name: 'single-patient', params: { patientID: get(['subject', 'reference'], selectedStatement) }}">
              {{ get(['subject', 'reference'], selectedStatement) }}
            </router-link>
          </td>
        </tr>
      </tbody>
    </table>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters } = createNamespacedHelpers('statement')

  export default {
    name: 'SingleStatementView',
    computed: mapGetters(['selectedStatement']),
    mounted () {
      this.statementID = this.$route.params.statementID
      this.$store.dispatch('statement/getSingleStatement', this.statementID)
      .then(data => {
        mapGetters(['selectedStatement'])
      })
    },
    methods: {
      taken (status) {
        switch (status) {
          case 'y':
            return 'yes'
          case 'n':
            return 'no'
          default:
            return 'unknown'
        }
      },
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      }
    }
  }
</script>
