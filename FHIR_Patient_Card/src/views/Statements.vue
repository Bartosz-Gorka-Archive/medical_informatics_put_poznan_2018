<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Medication Statements list</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">Medication Statements</li>
          </ol>
        </div>
      </div>
    </header>

      <table class="table table--data">
        <thead>
          <tr>
            <th class="u-hiddenDown@md">No</th>
            <th>ID</th>
            <th>Taken</th>
            <th>Dosage</th>
            <th>Concept</th>
            <th>Subject</th>
          </tr>
        </thead>
        <tfoot>
          <tr>
            <th class="u-hiddenDown@md">No</th>
            <th>ID</th>
            <th>Taken</th>
            <th>Dosage</th>
            <th>Concept</th>
            <th>Subject</th>
          </tr>
        </tfoot>
        <tbody>
          <template v-for="(statement, index) in this.statements">
            <tr>
              <td data-label="No" class="u-hiddenDown@md">{{ index + 1 }}</td>
              <td data-label="ID">{{ get(['resource', 'id'], statement) }}</td>
              <td data-label="Taken">{{ taken(get(['resource', 'taken'], statement)) }}</td>
              <td data-label="Dosage">{{ get(['resource', 'dosage', 0, 'text'], statement) }}</td>
              <td data-label="Concept" v-if="get(['resource', 'medicationCodeableConcept', 'text'], statement)">
                {{ get(['resource', 'medicationCodeableConcept', 'text'], statement) }}
              </td>
              <td data-label="Concept" v-else-if="get(['resource', 'medicationCodeableConcept', 'coding', 0, 'display'], statement)">
                {{ get(['resource', 'medicationCodeableConcept', 'coding', 0, 'display'], statement) }}
              </td>
              <td data-label="Concept" v-else-if="get(['resource', 'medicationReference', 'reference'], statement)">
                <router-link :to="{ name: 'single-medication', params: { medicationID: get(['resource', 'medicationReference', 'reference'], statement) }}">{{ get(['resource', 'medicationReference', 'display'], statement) }}</router-link>
              </td>
              <td data-label="Concept" v-else></td>
              <td data-label="Subject" v-if="get(['resource', 'subject', 'reference'], statement)">
                <router-link :to="{ name: 'single-patient', params: { patientID: get(['resource', 'subject', 'reference'], statement) }}">{{ get(['resource', 'subject', 'reference'], statement) }}</router-link>
              </td>
              <td data-label="Subject" v-else></td>
            </tr>
          </template>

        <infinite-loading
          v-if="loadingStatements"
          v-on:infinite="infiniteHandler"
          ref="infiniteLoading"
          spinner="bubbles">
        </infinite-loading>

        </tbody>
      </table>
    </div>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters, mapActions } = createNamespacedHelpers('statement')

  export default {
    name: 'StatementsView',
    computed: mapGetters(['loadingStatements', 'statements']),
    mounted () {
      mapActions(['clear'])
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
      },
      infiniteHandler (state) {
        this.$store.dispatch('statement/getStatements')
        .then(data => {
          mapGetters(['loadingStatements', 'statements'])
          if (this.loadingStatements) {
            state.loaded()
          } else {
            state.complete()
          }
        })
      }
    }
  }
</script>
